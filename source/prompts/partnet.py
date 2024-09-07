
# from partnet part2id
dict_coarse_grained_parts2id = {'chair': {'chair_head': '2',
  'chair_back': '5',
  'chair_arm': '16',
  'chair_base': '23',
  'chair_seat': '45'},
 'hat': {'crown': '3', 'brim': '4', 'button': '6', 'bill': '7', 'panel': '8'},
 'clock': {'clock_body': '3',
  'base': '7',
  'pendulum_clock_base': '13',
  'pendulum_clock_frame': '18',
  'pendulum': '20'},
 'pot': {'body': '2', 'base': '5', 'containing_things': '8'},
 'refrigerator': {'body': '2', 'base': '9'},
 'bottle': {'body': '10', 'lid': '5', 'handle': '11'},
 'bowl': {'container': '2', 'containing_things': '3', 'bottom': '4'},
 'bag': {'bag_body': '2', 'handle': '3', 'shoulder_strap': '4'},
 'dishwasher': {'body': '2', 'base': '7'},
 'table': {'ping_pong_net': '4',
  'tabletop': '30',
  'table_base': '38',
  'bench': '28'},
 'faucet': {'switch': '14', 'hose': '4', 'spout': '12', 'frame': '15'},
 'mug': {'body': '2', 'handle': '3', 'containing_things': '4'},
 'laptop': {'screen_side': '2', 'base_side': '5'},
 'cuttinginstrument': {'handle_side': '10', 'blade_side': '14'},
 'lamp': {'lamp_unit_group': '4',
  'lamp_body': '55',
  'chain': '12',
  'lamp_base': '63',
  'power_cord': '29',
  'pendant_lamp_unit': '19',
  'lamp_unit': '69',
  'lamp_post': '67',
  'street_lamp_base': '68'},
 'bed': {'bed_sleep_area': '3', 'bed_frame': '7', 'ladder': '22'},
 'earphone': {'earbud_unit': '3',
  'earbud_connector_wire': '7',
  'head_band': '9',
  'earcup_unit': '11',
  'connector_wire': '15'},
 'storagefurniture': {'countertop': '3',
  'shelf': '4',
  'cabinet_frame': '5',
  'drawer': '14',
  'cabinet_base': '21',
  'cabinet_door': '33'},
 'scissors': {'blade': '4', 'handle': '5'},
 'microwave': {'body': '2', 'base': '10'},
 'trashcan': {'container': '2',
  'outside_frame': '6',
  'base': '11',
  'cover': '13'},
 'door': {'outside_frame': '2', 'door': '3'},
 'keyboard': {'frame': '2', 'key': '3'},
 'display': {'display_screen': '2', 'base': '5'},
  'knife': {'handle_side': '3', 'blade_side': '7', 'handle_side': '10', 'blade_side': '14'},
  'vase': {'body': '2', 'base': '5', 'containing_things': '8'},
 }


# from partnet id2part
dict_coarse_grained_id2parts = {'chair': {'2': 'chair_head',
  '5': 'chair_back',
  '16': 'chair_arm',
  '23': 'chair_base',
  '45': 'chair_seat'},
 'hat': {'3': 'crown', '4': 'brim', '6': 'button', '7': 'bill', '8': 'panel'},
 'clock': {'3': 'clock_body',
  '7': 'base',
  '13': 'pendulum_clock_base',
  '18': 'pendulum_clock_frame',
  '20': 'pendulum'},
 'pot': {'2': 'body', '5': 'base', '8': 'containing_things'},
 'refrigerator': {'2': 'body', '9': 'base'},
 'bottle': {'3': 'body',
  '5': 'lid',
  '6': 'handle',
  '10': 'body',
  '11': 'handle'},
 'bowl': {'2': 'container', '3': 'containing_things', '4': 'bottom'},
 'bag': {'2': 'bag_body', '3': 'handle', '4': 'shoulder_strap'},
 'dishwasher': {'2': 'body', '7': 'base'},
 'table': {'4': 'ping_pong_net',
  '5': 'tabletop',
  '7': 'table_base',
  '13': 'tabletop',
  '17': 'table_base',
  '22': 'tabletop',
  '24': 'table_base',
  '28': 'bench',
  '30': 'tabletop',
  '38': 'table_base'},
 'faucet': {'3': 'switch',
  '4': 'hose',
  '5': 'spout',
  '7': 'frame',
  '12': 'spout',
  '14': 'switch',
  '15': 'frame'},
 'mug': {'2': 'body', '3': 'handle', '4': 'containing_things'},
 'laptop': {'2': 'screen_side', '5': 'base_side'},
 'cuttinginstrument': {'3': 'handle_side',
  '7': 'blade_side',
  '10': 'handle_side',
  '14': 'blade_side'},
 'lamp': {'4': 'lamp_unit_group',
  '11': 'lamp_body',
  '12': 'chain',
  '13': 'lamp_base',
  '17': 'power_cord',
  '19': 'pendant_lamp_unit',
  '25': 'lamp_base',
  '29': 'power_cord',
  '31': 'lamp_body',
  '38': 'lamp_unit',
  '49': 'lamp_base',
  '55': 'lamp_body',
  '56': 'lamp_unit',
  '63': 'lamp_base',
  '67': 'lamp_post',
  '68': 'street_lamp_base',
  '69': 'lamp_unit'},
 'bed': {'3': 'bed_sleep_area', '7': 'bed_frame', '22': 'ladder'},
 'earphone': {'3': 'earbud_unit',
  '7': 'earbud_connector_wire',
  '9': 'head_band',
  '11': 'earcup_unit',
  '15': 'connector_wire'},
 'storagefurniture': {'3': 'countertop',
  '4': 'shelf',
  '5': 'cabinet_frame',
  '14': 'drawer',
  '21': 'cabinet_base',
  '33': 'cabinet_door'},
 'scissors': {'4': 'blade', '5': 'handle'},
 'microwave': {'2': 'body', '10': 'base'},
 'trashcan': {'2': 'container',
  '6': 'outside_frame',
  '11': 'base',
  '13': 'cover'},
 'door': {'2': 'outside_frame', '3': 'door'},
 'keyboard': {'2': 'frame', '3': 'key'},
 'display': {'2': 'display_screen', '5': 'base'},
  'knife': {'3': 'handle_side', '7': 'blade_side', '10': 'handle_side', '14': 'blade_side'},
  'vaase': {'2': 'body', '5': 'base', '8': 'containing_things'},
 }

# for each object, the parts are ordered by the order of the partnet id
objects_parts_partnet_coarse = {
    'chair': ['chair_head',
                'chair_back',
                'chair_arm',
                'chair_base',
                'chair_seat'],
    'hat': ['crown', 'brim', 'button', 'bill', 'panel'],
    'clock': ['clock_body',
                'base',
                'pendulum_clock_base',
                'pendulum_clock_frame',
                'pendulum'],
    'pot': ['body', 'base', 'containing_things'],
    'refrigerator': ['body', 'base'],
    'bottle': ['body', 'lid', 'handle', 'body', 'handle'],
    'bowl': ['container', 'containing_things', 'bottom'],
    'bag': ['bag_body', 'handle', 'shoulder_strap'],
    'dishwasher': ['body', 'base'],
    'table': ['ping_pong_net',
                'tabletop',
                'table_base',
                'tabletop',
                'table_base',
                'tabletop',
                'table_base',
                'bench',
                'tabletop',
                'table_base'],
    'faucet': ['switch', 'hose', 'spout', 'frame', 'spout', 'switch', 'frame'],
    'mug': ['body', 'handle', 'containing_things'],
    'laptop': ['screen_side', 'base_side'],
    'cuttinginstrument': ['handle_side',
                            'blade_side',
                            'handle_side',
                            'blade_side'],
    'lamp': ['lamp_unit_group',
                'lamp_body',
                'chain',
                'lamp_base',
                'power_cord',
                'pendant_lamp_unit',
                'lamp_base',
                'power_cord',
                'lamp_body',
                'lamp_unit',
                'lamp_base',
                'lamp_body',
                'lamp_unit',
                'lamp_base',
                'lamp_post',
                'street_lamp_base',
                'lamp_unit'],
    'bed': ['bed_sleep_area', 'bed_frame', 'ladder'],
    'earphone': ['earbud_unit',
                    'earbud_connector_wire',
                    'head_band',
                    'earcup_unit',
                    'connector_wire'],
    'storagefurniture': ['countertop',
                    'shelf',
                    'cabinet_frame',
                    'drawer',
                    'cabinet_base',
                    'cabinet_door'],
    'scissors': ['blade', 'handle'],
    'microwave': ['body', 'base'],
    'trashcan': ['container', 'outside_frame', 'base', 'cover'],
    'door': ['outside_frame', 'door'],
    'keyboard': ['frame', 'key'],
    'display': ['display_screen', 'base'],
    'knife': ['handle_side', 'blade_side', 'handle_side', 'blade_side'],
    'vase': ['body', 'base', 'containing_things']
 }


dict_coarse_grained_parts_with_manual_annotations = {
  'chair': {
      'chair_head': 'The head of the chair is the top part of the chair',
      'chair_back': 'The back of the chair is the part where the person leans on',
      'chair_arm': 'The arm of the chair is the part where the person rests the arms',
      'chair_base': 'The base of the chair is the part where the chair is supported',
      'chair_seat': 'The seat of the chair is the part where the person sits'
  },
  'hat': {
      'crown': 'The crown of the hat is the top part of the hat',
      'brim': 'The brim of the hat is the part that extends outwards',
      'button': 'The button of the hat is the part that holds the hat together',
      'bill': 'The bill of the hat is the part that extends outwards',
      'panel': 'The panel of the hat is the part that holds the hat together'
  },
  'clock': {
      'clock_body': 'The body of the clock is the main part of the clock',
      'base': 'The base of the clock is the part where the clock is supported',
      'pendulum_clock_base': 'The pendulum clock base is the part where the pendulum is supported',
      'pendulum_clock_frame': 'The pendulum clock frame is the part where the pendulum is supported',
      'pendulum': 'The pendulum is the part that swings'
  },
  'pot': {
      'body': 'The body of the pot is the main part of the pot',
      'base': 'The base of the pot is the part where the pot is supported',
      'containing_things': 'The containing things of the pot is the part where the pot contains things'
  },
  'refrigerator': {
      'body': 'The body of the refrigerator is the main part of the refrigerator',
      'base': 'The base of the refrigerator is the part where the refrigerator is supported'
  },
  'bottle': {
      'body': 'The body of the bottle is the main part of the bottle',
      'lid': 'The lid of the bottle is the part that closes the bottle',
      'handle': 'The handle of the bottle is the part where the person holds the bottle'
  },
  'bowl': {
      'container': 'The container of the bowl is the main part of the bowl',
      'containing_things': 'The containing things of the bowl is the part where the bowl contains things',
      'bottom': 'The bottom of the bowl is the part where the bowl is supported'
  },
  'bag': {
      'bag_body': 'The bag body is the main part of the bag',
      'handle': 'The handle of the bag is the part where the person holds the bag',
      'shoulder_strap': 'The shoulder strap of the bag is the part where the person holds the bag'
  },
  'dishwasher': {
      'body': 'The body of the dishwasher is the main part of the dishwasher',
      'base': 'The base of the dishwasher is the part where the dishwasher is supported'
  },
  'table': {
      'ping_pong_net': 'The ping pong net is the part that separates the table',
      'tabletop': 'The tabletop is the main part of the table',
      'table_base': 'The table base is the part where the table is supported',
      'bench': 'The bench is the part where the person sits'
  },
  'faucet': {
      'switch': 'The switch is the part that turns the faucet on and off',
      'hose': 'The hose is the part that carries the water',
      'spout': 'The spout is the part where the water comes out',
      'frame': 'The frame is the main part of the faucet'
  },
  'mug': {
      'body': 'The body of the mug is the main part of the mug',
      'handle': 'The handle of the mug is the part where the person holds the mug',
      'containing_things': 'The containing things of the mug is the part where the mug contains things'
  },
  'laptop': {
      'screen_side': 'The screen side is the part where the screen is',
      'base_side': 'The base side is the main part of the laptop'
  },
  'cuttinginstrument': {
      'handle_side': 'The handle side is the part where the person holds the cutting instrument',
      'blade_side': 'The blade side is the part where the cutting instrument cuts'
  },
  'lamp': {
      'lamp_unit_group': 'The lamp unit group is the main part of the lamp',
      'lamp_body': 'The lamp body is the main part of the lamp',
      'chain': 'The chain is the part that holds the lamp',
      'lamp_base': 'The lamp base is the part where the lamp is supported',
      'power_cord': 'The power cord is the part that carries the electricity',
      'pendant_lamp_unit': 'The pendant lamp unit is the main part of the lamp',
      'lamp_unit': 'The lamp unit is the main part of the lamp',
      'lamp_post': 'The lamp post is the main part of the lamp',
      'street_lamp_base': 'The street lamp base is the part where the lamp is supported'
  },
  'bed': {
      'bed_sleep_area': 'The bed sleep area is the main part of the bed',
      'bed_frame': 'The bed frame is the part where the bed is supported',
      'ladder': 'The ladder is the part where the person climbs'
  },
  'earphone': {
      'earbud_unit': 'The earbud unit is the main part of the earphone',
      'earbud_connector_wire': 'The earbud connector wire is the part that connects the earbud',
      'head_band': 'The head band is the part where the person wears the earphone',
      'earcup_unit': 'The earcup unit is the main part of the earphone',
      'connector_wire': 'The connector wire is the part that connects the earphone'
  },
  'storagefurniture': {
      'countertop': 'The countertop is the main part of the storage furniture',
      'shelf': 'The shelf is the part where the things are placed',
      'cabinet_frame': 'The cabinet frame is the main part of the storage furniture',
      'drawer': 'The drawer is the part where the things are placed',
      'cabinet_base': 'The cabinet base is the part where the storage furniture is supported',
      'cabinet_door': 'The cabinet door is the part that opens and closes'
  },
  'scissors': {
      'blade': 'The blade is the part where the scissors cut',
      'handle': 'The handle is the part where the person holds the scissors'
  },
  'microwave': {
      'body': 'The body of the microwave is the main part of the microwave',
      'base': 'The base of the microwave is the part where the microwave is supported'
  },
  'trashcan': {
      'container': 'The container is the main part of the trash can',
      'outside_frame': 'The outside frame is the part where the trash can is supported',
      'base': 'The base is the part where the trash can is supported',
      'cover': 'The cover is the part that opens and closes'
  },
  'door': {
      'outside_frame': 'The outside frame is the part where the door is supported',
      'door': 'The door is the main part of the door'
  },
  'keyboard': {
      'frame': 'The frame is the main part of the keyboard',
      'key': 'The key is the part where the person presses'
  },
  'display': {
      'display_screen': 'The display screen is the main part of the display',
      'base': 'The base is the part where the display is supported'
  },
  'knife': {
      'handle_side': 'The handle side is the part where the person holds the knife',
      'blade_side': 'The blade side is the part where the knife cuts'
  },
  'vase': {
      'body': 'The body of the vase is the main part of the vase',
      'base': 'The base of the vase is the part where the vase is supported',
      'containing_things': 'The containing things of the vase is the part where the vase contains things'
  }   
}

dict_coarse_grained_parts_with_gpt3_annotations = {
  "chair": {
    "chair_head": "The chair head refers to the topmost part of the chair where the backrest meets the seat.",
    "chair_back": "The chair back is the vertical support behind the seating area, providing support to the back.",
    "chair_arm": "The chair arm is the horizontal part attached to the sides of the chair, providing support for the arms.",
    "chair_base": "The chair base is the foundation or support structure upon which the entire chair is built.",
    "chair_seat": "The chair seat is the horizontal surface where a person sits, usually positioned atop the chair base."
  },
  "hat": {
    "crown": "The crown of the hat is the top portion that covers the head, providing shape and structure.",
    "brim": "The brim is the horizontal edge that extends outward from the crown, providing shade and protection.",
    "button": "The button is a decorative or functional piece located at the top center of the hat, often used to attach accessories or hold parts together.",
    "bill": "The bill, also known as the visor, is the protruding, usually stiff, front portion of the hat that provides shade.",
    "panel": "The panel refers to any individual section of the hat's construction, usually stitched together to form the crown."
  },
  "clock": {
    "clock_body": "The clock body is the main housing that contains the clock's mechanism and displays the time.",
    "base": "The base of the clock is the bottom support structure upon which the entire clock stands.",
    "pendulum_clock_base": "The pendulum clock base is the lower portion of a pendulum clock that provides stability and support.",
    "pendulum_clock_frame": "The pendulum clock frame is the framework that holds and suspends the pendulum mechanism.",
    "pendulum": "The pendulum is a weighted rod or disk suspended from a pivot, which regulates the clock's movement."
  },
  "pot": {
    "body": "The pot body is the main container, usually cylindrical or round, used for holding or cooking substances.",
    "base": "The pot base is the bottom surface that provides stability and support to the entire pot structure.",
    "containing_things": "The containing things of the pot refer to any items or substances placed within the pot for storage or cooking purposes."
  },
  "refrigerator": {
    "body": "The refrigerator body is the main compartment that houses the cooling mechanism and shelves for storing food.",
    "base": "The refrigerator base is the bottom part of the appliance that provides support and stability."
  },
  "bottle": {
    "body": "The bottle body is the main container, usually cylindrical or flask-shaped, used for holding liquids or substances.",
    "lid": "The lid is the removable or hinged cover that seals the opening of the bottle to prevent spillage or contamination.",
    "handle": "The handle is a protruding part attached to the bottle, providing a grip for holding and pouring liquids."
  },
  "bowl": {
    "container": "The bowl container is the main concave portion used for holding food, liquids, or other substances.",
    "containing_things": "The containing things of the bowl refer to any items or substances placed within the bowl for storage or serving purposes.",
    "bottom": "The bottom of the bowl is the flat or curved surface that provides stability and support to the entire bowl structure."
  },
  "bag": {
    "bag_body": "The bag body is the main compartment or pouch used for holding items, typically made of fabric or leather.",
    "handle": "The handle is a strap or loop attached to the bag, designed for carrying it by hand.",
    "shoulder_strap": "The shoulder strap is a longer strap attached to the bag, designed for carrying it over the shoulder."
  },
  "dishwasher": {
    "body": "The dishwasher body is the main housing that contains the washing mechanism and racks for holding dishes.",
    "base": "The dishwasher base is the bottom part of the appliance that provides support and stability."
  },
  "table": {
    "ping_pong_net": "The ping pong net is the mesh barrier placed across the middle of the table to divide it for gameplay.",
    "tabletop": "The tabletop is the flat horizontal surface where items are placed or activities are conducted.",
    "table_base": "The table base is the support structure beneath the tabletop that holds the entire table together.",
    "bench": "The bench is a long seat without a backrest, typically placed alongside a table for seating."
  },
  "faucet": {
    "switch": "The switch is the lever or knob that controls the flow of water, allowing it to be turned on or off.",
    "hose": "The hose is the flexible tube that carries water from the plumbing to the spout, allowing for directed flow.",
    "spout": "The spout is the nozzle or outlet where water flows out of the faucet and into the sink or basin.",
    "frame": "The frame is the main structural support of the faucet, holding all the components together."
  },
  "mug": {
    "body": "The mug body is the main vessel used for holding and drinking beverages, typically cylindrical with a handle.",
    "handle": "The handle is the curved or looped appendage attached to the side of the mug, providing a grip for holding.",
    "containing_things": "The containing things of the mug refer to any liquids or substances held within the mug for consumption."
  },
  "laptop": {
    "screen_side": "The screen side is the part of the laptop where the display screen is located, typically facing the user.",
    "base_side": "The base side is the underside of the laptop housing the keyboard, ports, and internal components."
  },
  "cuttinginstrument": {
    "handle_side": "The handle side is the portion of the cutting instrument designed for gripping and manipulation.",
    "blade_side": "The blade side is the sharpened or cutting portion of the instrument used for slicing or shearing."
  },
  "lamp": {
    "lamp_unit_group": "The lamp unit group refers to the collection of components that make up the entire lamp assembly.",
    "lamp_body": "The lamp body is the main structure housing the light bulb and electrical components.",
    "chain": "The chain is a series of connected links used for hanging or suspending the lamp from a ceiling or fixture.",
    "lamp_base": "The lamp base is the bottom part of the lamp that provides stability and support.",
    "power_cord": "The power cord is the insulated electrical cable that connects the lamp to a power source.",
    "pendant_lamp_unit": "The pendant lamp unit is a type of lamp that hangs from the ceiling, typically suspended by a cord or chain.",
    "lamp_unit": "The lamp unit is the entire assembly of components used for generating light, including the bulb, socket, and shade.",
    "lamp_post": "The lamp post is a tall vertical support structure that holds the lamp unit, often used in outdoor lighting fixtures.",
    "street_lamp_base": "The street lamp base is the foundation or support structure upon which the entire street lamp is built."
  },
  "bed": {
    "bed_sleep_area": "The bed sleep area is the main surface where a person lies down for rest or sleep.",
    "bed_frame": "The bed frame is the structural support system that holds the mattress and sleep area above the ground.",
    "ladder": "The ladder is a set of steps or rungs used for climbing onto or off of elevated beds, such as bunk beds or lofts."
  },
  "earphone": {
    "earbud_unit": "The earbud unit is the component of the earphone that houses the speaker and is inserted into the ear canal.",
    "earbud_connector_wire": "The earbud connector wire is the cable that connects the earbuds to the audio source.",
    "head_band": "The headband is the adjustable strap or band that connects the earpieces and rests on the user's head, providing support and stability.",
    "earcup_unit": "The earcup unit is the part of over-ear headphones that houses the speaker and covers the user's ears.",
    "connector_wire": "The connector wire is the cable that connects the earphones to the audio source, typically terminating in a plug."
  },
  "storagefurniture": {
    "countertop": "The countertop is the flat horizontal surface on top of storage furniture, often used for placing items or preparing food.",
    "shelf": "The shelf is a horizontal platform within storage furniture used for storing or organizing items.",
    "cabinet_frame": "The cabinet frame is the structural framework that forms the skeleton of the storage furniture, providing support and shape.",
    "drawer": "The drawer is a sliding compartment within storage furniture used for storing items, accessed by pulling it out from the cabinet.",
    "cabinet_base": "The cabinet base is the bottom part of storage furniture that provides support and stability, often raising it off the ground.",
    "cabinet_door": "The cabinet door is the hinged or sliding panel that covers the front of a storage cabinet, providing access to its contents."
  },
  "scissors": {
    "blade": "The blade is the sharpened cutting edge of the scissors used for cutting materials.",
    "handle": "The handle is the gripping portion of the scissors, usually made of plastic or metal, used for holding and operating the blades."
  },
  "microwave": {
    "body": "The body of the microwave is the main housing that contains the control panel, cooking chamber, and door.",
    "base": "The base of the microwave is the bottom surface that provides support and stability to the entire appliance."
  },
  "trashcan": {
    "container": "The container is the main receptacle of the trash can, used for holding waste materials.",
    "outside_frame": "The outside frame is the structural support system surrounding the container, providing stability and shape to the trash can.",
    "base": "The base is the bottom portion of the trash can that provides support and stability, typically in contact with the ground.",
    "cover": "The cover is the lid or top portion of the trash can that opens and closes to conceal or reveal the container."
  },
  "door": {
    "outside_frame": "The outside frame is the structural support system surrounding the door, providing stability and shape to the door set.",
    "door": "The door is the movable panel that opens and closes to allow passage through the doorway, typically hinged or sliding."
  },
  "keyboard": {
    "frame": "The frame is the structural support system of the keyboard, providing stability and shape to the keys and internal components.",
    "key": "The key is the individual button or switch on the keyboard that is pressed by the user to input characters or commands."
  },
  "display": {
    "display_screen": "The display screen is the flat panel or surface on which visual information is displayed, such as images or text.",
    "base": "The base is the bottom portion of the display that provides support and stability, typically resting on a desk or stand."
  },
  "knife": {
    "handle_side": "The handle side is the portion of the knife designed for gripping and manipulation.",
    "blade_side": "The blade side is the sharpened or cutting portion of the knife used for slicing or shearing."
  },
  "vase": {
    "body": "The body of the vase is the main container, usually cylindrical or urn-shaped, used for holding flowers or decorative items.",
    "base": "The base of the vase is the bottom surface that provides support and stability to the entire structure.",
    "containing_things": "The containing things of the vase refer to any items or substances placed within the vase for decorative purposes."
  }
}