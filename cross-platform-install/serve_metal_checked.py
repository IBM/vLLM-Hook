"""Run the vLLM CLI only when its selected platform is Metal."""
from vllm.platforms import current_platform
from vllm_metal.platform import MetalPlatform


def main():
    if not isinstance(current_platform, MetalPlatform):
        raise RuntimeError(
            "Metal is required; vLLM selected "
            f"{type(current_platform).__module__}.{type(current_platform).__name__}. "
            "Set VLLM_LOGGING_LEVEL=DEBUG to inspect plugin registration."
        )
    print("Verified platform: MetalPlatform", flush=True)
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    main()
