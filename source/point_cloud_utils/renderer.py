import pyrender
import numpy as np


def renderer(
        point_cloud,
        colors=None,
        camera_type='intrinsic',
        CANVAS_WIDTH=600, CANVAS_HEIGHT=600,
        FOV=np.pi / 6.2,
        FX=1000, FY=1000,
        CX=600/2, CY=600/2,
        point_size=5.0,
        light_intensity=3.0,
        interactive=False
):
    assert camera_type in ['intrinsic', 'perspective', 'orthographic'], 'Camera type is not valid'

    # ============ #
    # Scene setup
    # ============ #

    # Initialize the scene and add both an ambient light and a background color
    scene = pyrender.Scene()

    point_mesh = pyrender.Mesh.from_points(np.asarray(point_cloud.points), colors=colors)

    # Create a mesh node and add it to the scene
    node_point_cloud = pyrender.Node(mesh=point_mesh)
    scene.add_node(node_point_cloud)

    # ============ #
    # Camera setup
    # ============ #

    if camera_type == 'perspective':
        camera = pyrender.PerspectiveCamera(name='main_cam', yfov=FOV, aspectRatio=1)
    elif camera_type == 'intrinsic':
        camera = pyrender.IntrinsicsCamera(name='main_cam', fx=FX, fy=FY, cx=CX, cy=CY)
    else:
        camera = None

    # Camera pose
    camera_pose = np.eye(4)

    # Create the camera node and add it to the scene
    node_camera = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(node_camera)

    # ============ #
    # Lighting
    # ============ #

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    scene.add(light, pose=camera_pose)

    # ============ #
    # Rendering
    # ============ #

    renderer = pyrender.OffscreenRenderer(CANVAS_WIDTH, CANVAS_HEIGHT, point_size=point_size)

    if interactive:
        pyrender.Viewer(scene, point_size=point_size)

    render, depth = renderer.render(scene)
    return render, depth


def get_camera_matrix(FX, FY, CX, CY):
    """
    Instantiate a camera matrix given the intrinsic parameters

    :param FX: horizontal focal length
    :param FY: vertical focal length
    :param CX: horizontal displacement
    :param CY: vertical displacement
    :return:
    """

    return np.array([
        [FX, 0.0, CX, 0.0],
        [0.0, FY, CY, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
