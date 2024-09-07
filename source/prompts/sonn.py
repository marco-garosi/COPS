id2category = {
    0: 'bag',
    1: 'bin',
    2: 'box',
    3: 'cabinet',
    4: 'chair',
    5: 'desk',
    6: 'display',
    7: 'door',
    8: 'shelf',
    9: 'table',
    10: 'bed',
    11: 'pillow',
    12: 'sink',
    13: 'sofa',
    14: 'toilet',
}

classification_descriptions = {
    'bag': 'A bag is a flexible container with a single opening.',
    'bin': 'A bin is a container for waste.',
    'box': 'A box is a container with a flat base and sides.',
    'cabinet': 'A cabinet is a cupboard with shelves or drawers for storing or displaying articles.',
    'chair': 'A chair is a separate seat for one person, typically with a back and four legs.',
    'desk': 'A desk is a piece of furniture with a flat or sloped surface for writing or working at.',
    'display': 'A display is a computer screen or other device for showing text and images.',
    'door': 'A door is a hinged, sliding, or revolving barrier at the entrance to a building, room, or vehicle.',
    'shelf': 'A shelf is a flat length of wood or rigid material, attached to a wall or forming part of a piece of furniture, that provides a surface for the storage or display of objects.',
    'table': 'A table is a piece of furniture with a flat top and one or more legs, providing a level surface for eating, writing, or working at.',
    'bed': 'A bed is a piece of furniture for sleep or rest, typically a framework with a mattress and coverings.',
    'pillow': 'A pillow is a rectangular cloth bag stuffed with feathers, foam rubber, or other soft materials, used to support the head when lying down.',
    'sink': 'A sink is a fixed basin with a water supply and outflow pipe.',
    'sofa': 'A sofa is a long upholstered seat with a back and arms, for two or more people.',
    'toilet': 'A toilet is a fixed receptacle into which a person may urinate or defecate, typically consisting of a bowl and a seat.'
}


objects_with_partsDescriptions_background = {
    'bag': {  # category 0
        'unknown': 'the background of the bag',
        'bag_body': 'the main part of the bag',
        'handle': 'the handle of the bag',  # not in test set
        'shoulder_strap': 'the shoulder strap of the bag'
    },
    'bin': {  # category 1
        'unknown': 'the background of the bin',
        'container': 'the main part of the bin',
        'outside_frame': 'the outside frame of the bin',  # not in test set
        'base': 'the base of the bin',  # never found
        'cover': 'the cover of the bin'
    },
    'box': {  # category 2
        'unknown': 'the background of the box',
        'container': 'the main part of the box',
        'containing_things': 'the part of the box containing things',
        'bottom': 'the bottom of the box',
        'cover': 'the cover of the box'
    },
    'cabinet': {   # category 3
        'unknown': 'the background of the cabinet',
        'countertop': 'the countertop of the cabinet',
        'shelf': 'the shelf of the cabinet',
        'frame': 'the frame of the cabinet',
        'drawer': 'the drawer of the cabinet',
        'base': 'the base of the cabinet',
        'door': 'the door of the cabinet'
    },
    'chair': {  # category 4
        'unknown': 'the background of the chair',
        'head': 'the head of the chair',  # never found
        'back': 'the back of the chair',
        'arm': 'the arm of the chair',
        'base': 'the base of the chair',
        'seat': 'the seat of the chair'
    },
    'desk': {  # category 5
        'unknown': 'the background of the desk',
        'desktop': 'the desktop of the desk',
        'base': 'the base of the desk',
        'drawer': 'the drawer of the desk'
    },
    'display': {  # category 6
        'unknown': 'the background of the display',
        'screen': 'the screen of the display',
        'base': 'the base of the display'
    },
    'door': {  # category 7
        'unknown': 'the background of the door',
        'outside_frame': 'the outside frame of the door',
        'door': 'the door itself'
    },
    'shelf': {  # category 8
        'unknown': 'the background of the shelf',
        'frame': 'the frame of the shelf',
        'surface': 'the surface of the shelf',
        'thing': 'the thing on the shelf'
    },
    'table': {  # category 9
        'unknown': 'the background of the table',
        'tabletop': 'the tabletop of the table',
        'base': 'the base of the table'
    },
    'bed': {  # category 10
        'unknown': 'the background of the bed',
        'sleep_area': 'the sleep area of the bed',
        'frame': 'the frame of the bed'
    },
    'pillow': {  # category 11
        'unknown': 'the background of the pillow',
        'pillow': 'the pillow itself'
    },
    'sink': {  # category 12
        'unknown': 'the background of the sink',
        'bowl': 'the bowl of the sink',
        'faucet': 'the faucet of the sink',
        'drain': 'the drain of the sink'
    },
    'sofa': {  # category 13
        'unknown': 'the background of the sofa',
        'back': 'the back of the sofa',
        'arm': 'the arm of the sofa',
        'base': 'the base of the sofa',
        'seat': 'the seat of the sofa'
    },
    'toilet': {  # category 14
        'unknown': 'the background of the toilet',
        'tank': 'the tank of the toilet',
        'bowl': 'the bowl of the toilet',
        'cover': 'the cover of the toilet',
        'seat': 'the seat of the toilet',
        'base': 'the base of the toilet'
    }
}

objects_with_part_descriptions_background_remapped = {
    'bag': {  # category 0
        'unknown': 'the background of the bag',
        'bag_body': 'the main part of the bag',
        'handle': 'the handle of the bag',  # not in test set
        'shoulder_strap': 'the shoulder strap of the bag'
    },
    'bin': {  # category 1
        'unknown': 'the background of the bin',
        'container': 'the main part of the bin',
        'outside_frame': 'the outside frame of the bin',  # not in test set
        'cover': 'the cover of the bin'
    },
    'box': {  # category 2
        'unknown': 'the background of the box',
        'container': 'the main part of the box',
        'containing_things': 'the part of the box containing things',
        'bottom': 'the bottom of the box',
        'cover': 'the cover of the box'
    },
    'cabinet': {   # category 3
        'unknown': 'the background of the cabinet',
        'countertop': 'the countertop of the cabinet',
        'shelf': 'the shelf of the cabinet',
        'frame': 'the frame of the cabinet',
        'drawer': 'the drawer of the cabinet',
        'base': 'the base of the cabinet',
        'door': 'the door of the cabinet'
    },
    'chair': {  # category 4
        'unknown': 'the background of the chair',
        'back': 'the back of the chair',
        'arm': 'the arm of the chair',
        'base': 'the base of the chair',
        'seat': 'the seat of the chair'
    },
    'desk': {  # category 5
        'unknown': 'the background of the desk',
        'desktop': 'the desktop of the desk',
        'base': 'the base of the desk',
        'drawer': 'the drawer of the desk'
    },
    'display': {  # category 6
        'unknown': 'the background of the display',
        'screen': 'the screen of the display',
        'base': 'the base of the display'
    },
    'door': {  # category 7
        'unknown': 'the background of the door',
        'outside_frame': 'the outside frame of the door',
        'door': 'the door itself'
    },
    'shelf': {  # category 8
        'unknown': 'the background of the shelf',
        'frame': 'the frame of the shelf',
        'surface': 'the surface of the shelf',
        'thing': 'the thing on the shelf'
    },
    'table': {  # category 9
        'unknown': 'the background of the table',
        'tabletop': 'the tabletop of the table',
        'base': 'the base of the table'
    },
    'bed': {  # category 10
        'unknown': 'the background of the bed',
        'sleep_area': 'the sleep area of the bed',
        'frame': 'the frame of the bed'
    },
    'pillow': {  # category 11
        'unknown': 'the background of the pillow',
        'pillow': 'the pillow itself'
    },
    'sink': {  # category 12
        'unknown': 'the background of the sink',
        'bowl': 'the bowl of the sink',
        'faucet': 'the faucet of the sink',
        'drain': 'the drain of the sink'
    },
    'sofa': {  # category 13
        'unknown': 'the background of the sofa',
        'back': 'the back of the sofa',
        'arm': 'the arm of the sofa',
        'base': 'the base of the sofa',
        'seat': 'the seat of the sofa'
    },
    'toilet': {  # category 14
        'unknown': 'the background of the toilet',
        'tank': 'the tank of the toilet',
        'bowl': 'the bowl of the toilet',
        'cover': 'the cover of the toilet',
        'seat': 'the seat of the toilet',
        'base': 'the base of the toilet'
    }
}

objects_with_partsDescriptions = {
    'bag': {  # Category 0
        'bag_body': 'the main part of the bag',
        'handle': 'the handle of the bag',
        'shoulder_strap': 'the shoulder strap of the bag'
    },
    'bin': {  # Category 1
        'container': 'the main part of the bin',
        'outside_frame': 'the outside frame of the bin',
        'base': 'the base of the bin',
        'cover': 'the cover of the bin'
    },
    'box': {  # Category 2
        'container': 'the main part of the box',
        'containing_things': 'the part of the box containing things',
        'bottom': 'the bottom of the box',
        'cover': 'the cover of the box'
    },
    'cabinet': {  # Category 3
        'countertop': 'the countertop of the cabinet',
        'shelf': 'the shelf of the cabinet',
        'frame': 'the frame of the cabinet',
        'drawer': 'the drawer of the cabinet',
        'base': 'the base of the cabinet',
        'door': 'the door of the cabinet'
    },
    'chair': {  # Category 4
        'head': 'the head of the chair',
        'back': 'the back of the chair',
        'arm': 'the arm of the chair',
        'base': 'the base of the chair',
        'seat': 'the seat of the chair'
    },
    'desk': {  # Category 0
        'desktop': 'the desktop of the desk',
        'base': 'the base of the desk',
        'drawer': 'the drawer of the desk'
    },
    'display': {  # Category 6
        'screen': 'the screen of the display',
        'base': 'the base of the display'
    },
    'door': {  # Category 7
        'outside_frame': 'the outside frame of the door',
        'door': 'the door itself'
    },
    'shelf': {  # Category 8
        'frame': 'the frame of the shelf',
        'surface': 'the surface of the shelf',
        'thing': 'the thing on the shelf'
    },
    'table': {  # Category 9
        'tabletop': 'the tabletop of the table',
        'base': 'the base of the table'
    },
    'bed': {  # Category 10
        'sleep_area': 'the sleep area of the bed',
        'frame': 'the frame of the bed'
    },
    'pillow': {  # Category 11
        'pillow': 'the pillow itself'
    },
    'sink': {  # Category 12
        'bowl': 'the bowl of the sink',
        'faucet': 'the faucet of the sink',
        'drain': 'the drain of the sink'
    },
    'sofa': {  # Category 13
        'back': 'the back of the sofa',
        'arm': 'the arm of the sofa',
        'base': 'the base of the sofa',
        'seat': 'the seat of the sofa'
    },
    'toilet': {  # Category 14
        'tank': 'the tank of the toilet',
        'bowl': 'the bowl of the toilet',
        'cover': 'the cover of the toilet',
        'seat': 'the seat of the toilet',
        'base': 'the base of the toilet'
    }
}

objects_with_part_descriptions_remapped = {
    'bag': {  # Category 0
        'bag_body': 'the main part of the bag',
        'handle': 'the handle of the bag',
        'shoulder_strap': 'the shoulder strap of the bag'
    },
    'bin': {  # Category 1
        'container': 'the main part of the bin',
        'outside_frame': 'the outside frame of the bin',
        'cover': 'the cover of the bin'
    },
    'box': {  # Category 2
        'container': 'the main part of the box',
        'containing_things': 'the part of the box containing things',
        'bottom': 'the bottom of the box',
        'cover': 'the cover of the box'
    },
    'cabinet': {  # Category 3
        'countertop': 'the countertop of the cabinet',
        'shelf': 'the shelf of the cabinet',
        'frame': 'the frame of the cabinet',
        'drawer': 'the drawer of the cabinet',
        'base': 'the base of the cabinet',
        'door': 'the door of the cabinet'
    },
    'chair': {  # Category 4
        'back': 'the back of the chair',
        'arm': 'the arm of the chair',
        'base': 'the base of the chair',
        'seat': 'the seat of the chair'
    },
    'desk': {  # Category 0
        'desktop': 'the desktop of the desk',
        'base': 'the base of the desk',
        'drawer': 'the drawer of the desk'
    },
    'display': {  # Category 6
        'screen': 'the screen of the display',
        'base': 'the base of the display'
    },
    'door': {  # Category 7
        'outside_frame': 'the outside frame of the door',
        'door': 'the door itself'
    },
    'shelf': {  # Category 8
        'frame': 'the frame of the shelf',
        'surface': 'the surface of the shelf',
        'thing': 'the thing on the shelf'
    },
    'table': {  # Category 9
        'tabletop': 'the tabletop of the table',
        'base': 'the base of the table'
    },
    'bed': {  # Category 10
        'sleep_area': 'the sleep area of the bed',
        'frame': 'the frame of the bed'
    },
    'pillow': {  # Category 11
        'pillow': 'the pillow itself'
    },
    'sink': {  # Category 12
        'bowl': 'the bowl of the sink',
        'faucet': 'the faucet of the sink',
        'drain': 'the drain of the sink'
    },
    'sofa': {  # Category 13
        'back': 'the back of the sofa',
        'arm': 'the arm of the sofa',
        'base': 'the base of the sofa',
        'seat': 'the seat of the sofa'
    },
    'toilet': {  # Category 14
        'tank': 'the tank of the toilet',
        'bowl': 'the bowl of the toilet',
        'cover': 'the cover of the toilet',
        'seat': 'the seat of the toilet',
        'base': 'the base of the toilet'
    }
}
