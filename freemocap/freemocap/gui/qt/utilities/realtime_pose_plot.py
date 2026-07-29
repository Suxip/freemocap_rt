HORIZONTAL_AND_DEPTH_TICKS = (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
HEIGHT_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)


def configure_realtime_pose_axes(axes, title: str) -> None:
    """Apply the identical front-facing view and scale to both RT pose plots."""
    axes.set_title(title)
    axes.set_proj_type("ortho")
    axes.view_init(elev=0, azim=-90, roll=0)
    axes.set_xlim(-0.75, 0.75)
    axes.set_ylim(-0.75, 0.75)
    axes.set_zlim(0.0, 1.5)
    axes.set_xticks(HORIZONTAL_AND_DEPTH_TICKS)
    axes.set_yticks(HORIZONTAL_AND_DEPTH_TICKS)
    axes.set_zticks(HEIGHT_TICKS)
    axes.set_xlabel("X")
    axes.set_ylabel("Depth")
    axes.set_zlabel("Height")
