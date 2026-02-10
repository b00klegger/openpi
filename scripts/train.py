import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if config.gradient_checkpointing:
            # Capture train=True in closure so it's a Python constant, not a traced value.
            # jax.checkpoint traces all args, and compute_loss uses `if train:` which
            # requires a concrete (non-traced) boolean.
            chunked_loss = jax.checkpoint(lambda r, o, a: model.compute_loss(r, o, a, train=True))(
                rng, observation, actions
            )
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _get_per_device_memory_gb() -> float | None:
    """Returns per-device memory in GiB, or None if it cannot be determined."""
    try:
        devices = jax.devices()
        if not devices:
            return None
        mem_stats = devices[0].memory_stats()
        if mem_stats and "bytes_limit" in mem_stats:
            return mem_stats["bytes_limit"] / (1024**3)
    except Exception:
        pass
    return None


def _auto_configure_fsdp(config: _config.TrainConfig) -> _config.TrainConfig:
    """Auto-enable FSDP and scale batch size for multi-GPU setups with limited per-device memory.

    When fsdp_devices is at its default (1) and the system has multiple GPUs with
    less than 24 GiB each, FSDP is enabled across all devices to shard model parameters
    and the batch size is scaled down to fit per-device memory.

    Even with FSDP, peak memory during the forward pass includes temporarily all-gathered
    params plus activations, so batch size must also be reduced for small GPUs.
    """
    num_devices = jax.device_count()
    if num_devices <= 1 or config.fsdp_devices > 1:
        return config

    mem_gb = _get_per_device_memory_gb()
    if mem_gb is None:
        logging.warning(
            f"Multiple devices detected ({num_devices}) but could not determine per-device memory. "
            f"Consider setting --fsdp-devices={num_devices} and reducing --batch-size if you encounter OOM errors."
        )
        return config

    # 24 GiB threshold: models like pi0/pi05 need ~4-5 GiB for params alone in bf16,
    # plus optimizer states and activations. Devices with < 24 GiB benefit from FSDP.
    if mem_gb < 24.0:
        new_fsdp = num_devices
        logging.info(
            f"Auto-enabling FSDP: {mem_gb:.1f} GiB per device across {num_devices} devices. "
            f"Setting fsdp_devices={new_fsdp} to shard model parameters and reduce per-device memory."
        )

        # Consumer/prosumer GPUs (which typically have < 24 GiB) often lack proper
        # peer-to-peer (P2P) GPU memory access support (no NVLink). NCCL's default
        # P2P mode causes "illegal memory access" errors on these cards. Disable P2P
        # to force NCCL to use shared memory/host staging instead.
        if "NCCL_P2P_DISABLE" not in os.environ:
            os.environ["NCCL_P2P_DISABLE"] = "1"
            logging.info("Setting NCCL_P2P_DISABLE=1 for consumer GPU compatibility.")
        config = dataclasses.replace(config, fsdp_devices=new_fsdp, gradient_checkpointing=True)

        # Scale batch size for per-device memory. During FSDP forward pass, peak memory
        # includes temporarily all-gathered full params (~4-5 GiB) plus activations.
        # Target max 2 samples per device for GPUs <= 16 GiB, 4 for GPUs <= 20 GiB.
        if mem_gb <= 16.0:
            max_per_device = 2
        elif mem_gb <= 20.0:
            max_per_device = 4
        else:
            max_per_device = 8

        max_batch = max_per_device * num_devices
        if config.batch_size > max_batch:
            logging.info(
                f"Reducing batch_size from {config.batch_size} to {max_batch} "
                f"({max_per_device} samples/device) to fit in {mem_gb:.1f} GiB per-device memory."
            )
            config = dataclasses.replace(config, batch_size=max_batch)

        # Ensure batch size is still divisible by the total mesh size.
        if config.batch_size % num_devices != 0:
            new_batch_size = (config.batch_size // num_devices) * num_devices
            if new_batch_size == 0:
                new_batch_size = num_devices
            logging.info(
                f"Adjusting batch_size from {config.batch_size} to {new_batch_size} "
                f"for compatibility with {num_devices} devices."
            )
            config = dataclasses.replace(config, batch_size=new_batch_size)
    else:
        logging.info(
            f"FSDP not auto-enabled: {mem_gb:.1f} GiB per device is sufficient for parameter replication."
        )

    return config


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    num_devices = jax.device_count()
    mem_gb = _get_per_device_memory_gb()
    logging.info(
        f"Devices: {num_devices}, per-device memory: {f'{mem_gb:.1f} GiB' if mem_gb else 'unknown'}, "
        f"fsdp_devices: {config.fsdp_devices}"
    )

    config = _auto_configure_fsdp(config)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
