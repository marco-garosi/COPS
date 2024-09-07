objects_parts_partslip = {
 'Bottle': ['lid'],
 'Box': ['lid'],
 'Bucket': ['handle'],
 'Camera': ['button', 'lens'],
 'Cart': ['wheel'],
 'Chair': ['arm', 'back', 'leg', 'seat', 'wheel'],
 'Clock': ['hand'],
 'CoffeeMachine': ['button', 'container', 'knob', 'lid'],
 'Dishwasher': ['door', 'handle'],
 'Dispenser': ['head', 'lid'],
 'Display': ['base', 'screen', 'support'],
 'Door': ['frame', 'door', 'handle'],
 'Eyeglasses': ['body', 'leg'],
 'Faucet': ['spout', 'switch'],
 'FoldingChair': ['seat'],
 'Globe': ['sphere'],
 'Kettle': ['lid', 'handle', 'spout'],
 'Keyboard': ['cord', 'key'],
 'KitchenPot': ['lid', 'handle'],
 'Knife': ['blade'],
 'Lamp': ['base', 'body', 'bulb', 'shade'],
 'Laptop': ['keyboard', 'screen', 'shaft', 'touchpad', 'camera'],
 'Lighter': ['lid', 'wheel', 'button'],
 'Microwave': ['display', 'door', 'handle', 'button'],
 'Mouse': ['button', 'cord', 'wheel'],
 'Oven': ['door', 'knob'],
 'Pen': ['cap', 'button'],
 'Phone': ['lid', 'button'],
 'Pliers': ['leg'],
 'Printer': ['button'],
 'Refrigerator': ['door', 'handle'],
 'Remote': ['button'],
 'Safe': ['door', 'switch', 'button'],
 'Scissors': ['blade', 'handle', 'screw'],
 'Stapler': ['body', 'lid'],
 'StorageFurniture': ['door', 'drawer', 'handle'],
 'Suitcase': ['handle', 'wheel'],
 'Switch': ['switch'],
 'Table': ['door', 'drawer', 'leg', 'tabletop', 'wheel', 'handle'],
 'Toaster': ['button', 'slider'],
 'Toilet': ['lid', 'seat', 'button'],
 'TrashCan': ['footpedal', 'lid', 'door'],
 'USB': ['cap', 'rotation'],
 'WashingMachine': ['door', 'button'],
 'Window': ['window']
 }

id2objectcategory = {idx: obj for idx, (obj, its) in enumerate(objects_parts_partslip.items())}



objects_parts_partslip_with_parts_description_v1 = {
    'Bottle': {
        'lid': 'The lid of a bottle is the top part that can be removed to access the contents inside.'
    },
    'Box': {
        'lid': 'The lid of a box is the top part that can be removed to access the contents inside.'
    },
    'Bucket': {
        'handle': 'The handle of a bucket is the part that is used to carry the bucket.'
    },
    'Camera': {
        'button': 'A button is a small, usually circular object that is used to control the camera.',
        'lens': 'The lens of a camera is the part that focuses the light onto the film or sensor.'
    },
    'Cart': {
        'wheel': 'A wheel is a circular object that is used to move the cart.'
    },
    'Chair': {
        'arm': 'The arm of a chair is the part that you rest your arms on.',
        'back': 'The back of a chair is the part that you lean against.',
        'leg': 'The leg of a chair is the part that supports the chair.',
        'seat': 'The seat of a chair is the part that you sit on.',
        'wheel': 'A wheel is a circular object that is used to move the chair.'
    },
    'Clock': {
        'hand': 'The hands of a clock are the parts that move to show the time.'
    },
    'CoffeeMachine': {
        'button': 'A button is a small, usually circular object that is used to control the coffee machine.',
        'container': 'The container of a coffee machine is the part that holds the coffee grounds and water.',
        'knob': 'A knob is a small, usually circular object that is used to control the coffee machine.',
        'lid': 'The lid of a coffee machine is the top part that can be removed to access the contents inside.'
    },
    'Dishwasher': {
        'door': 'The door of a dishwasher is the part that you open to put dishes inside.',
        'handle': 'The handle of a dishwasher is the part that you use to open the door.'
    },
    'Dispenser': {
        'head': 'The head of a dispenser is the part that dispenses the contents.',
        'lid': 'The lid of a dispenser is the top part that can be removed to access the contents inside.'
    },
    'Display': {
        'base': 'The base of a display is the part that supports the screen.',
        'screen': 'The screen of a display is the part that shows the images.',
        'support': 'The support of a display is the part that holds the screen.'
    },
    'Door': {
        'frame': 'The frame of a door is the part that holds the door in place.',
        'door': 'The door of a door is the part that you open to go through.',
        'handle': 'The handle of a door is the part that you use to open the door.'
    },
    'Eyeglasses': {
        'body': 'The body of eyeglasses is the part that holds the lenses.',
        'leg': 'The leg of eyeglasses is the part that goes over your ears.'
    },
    'Faucet': {
        'spout': 'The spout of a faucet is the part that the water comes out of.',
        'switch': 'A switch is a small, usually circular object that is used to control the faucet.'
    },
    'FoldingChair': {
        'seat': 'The seat of a folding chair is the part that you sit on.'
    },
    'Globe': {
        'sphere': 'The sphere of a globe is the part that shows the earth.'
    },
    'Kettle': {
        'lid': 'The lid of a kettle is the top part that can be removed to access the contents inside.',
        'handle': 'The handle of a kettle is the part that you use to pour the water.',
        'spout': 'The spout of a kettle is the part that the water comes out of.'
    },
    'Keyboard': {
        'cord': 'The cord of a keyboard is the part that connects it to the computer.',
        'key': 'A key is a small, usually rectangular object that is used to type on the keyboard.'
    },
    'KitchenPot': {
        'lid': 'The lid of a kitchen pot is the top part that can be removed to access the contents inside.',
        'handle': 'The handle of a kitchen pot is the part that you use to pour the contents.',
    },
    'Knife': {
        'blade': 'The blade of a knife is the part that cuts.',
        'handle': 'The handle of a knife is the part that you hold.'
    },
    'Lamp': {
        'base': 'The base of a lamp is the part that supports the body.',
        'body': 'The body of a lamp is the part that holds the bulb.',
        'bulb': 'The bulb of a lamp is the part that lights up.',
        'shade': 'The shade of a lamp is the part that covers the bulb.'
    },
    'Laptop': {
        'keyboard': 'The keyboard of a laptop is the part that you type on.',
        'screen': 'The screen of a laptop is the part that shows the images.',
        'shaft': 'The shaft of a laptop is the part that holds the screen.',
        'touchpad': 'The touchpad of a laptop is the part that you use to control the cursor.',
        'camera': 'The camera of a laptop is the part that takes pictures.'
    },
    'Lighter': {
        'lid': 'The lid of a lighter is the top part that can be removed to access the contents inside.',
        'wheel': 'The wheel of a lighter is the part that creates the spark.',
        'button': 'A button is a small, usually circular object that is used to control the lighter.'
    },
    'Microwave': {
        'display': 'The display of a microwave is the part that shows the time and settings.',
        'door': 'The door of a microwave is the part that you open to put food inside.',
        'handle': 'The handle of a microwave is the part that you use to open the door.',
        'button': 'A button is a small, usually circular object that is used to control the microwave.'
    },
    'Mouse': {
        'button': 'A button is a small, usually circular object that is used to control the mouse.',
        'cord': 'The cord of a mouse is the part that connects it to the computer.',
        'wheel': 'The wheel of a mouse is the part that you use to scroll.'
    },
    'Oven': {
        'door': 'The door of an oven is the part that you open to put food inside.',
        'knob': 'A knob is a small, usually circular object that is used to control the oven.'
    },
    'Pen': {
        'cap': 'The cap of a pen is the top part that can be removed to access the writing tip.',
        'button': 'A button is a small, usually circular object that is used to control the pen.'
    },
    'Phone': {
        'lid': 'The lid of a phone is the top part that can be removed to access the contents inside.',
        'button': 'A button is a small, usually circular object that is used to control the phone.'
    },
    'Pliers': {
        'leg': 'The leg of pliers is the part that you hold.',
    },
    'Printer': {
        'button': 'A button is a small, usually circular object that is used to control the printer.'
    },
    'Refrigerator': {
        'door': 'The door of a refrigerator is the part that you open to put food inside.',
        'handle': 'The handle of a refrigerator is the part that you use to open the door.'
    },
    'Remote': {
        'button': 'A button is a small, usually circular object that is used to control the remote.'
    },
    'Safe': {
        'door': 'The door of a safe is the part that you open to access the contents inside.',
        'switch': 'A switch is a small, usually circular object that is used to control the safe.',
        'button': 'A button is a small, usually circular object that is used to control the safe.'
    },
    'Scissors': {
        'blade': 'The blade of scissors is the part that cuts.',
        'handle': 'The handle of scissors is the part that you hold.',
        'screw': 'The screw of scissors is the part that holds the two blades together.'
    },
    'Stapler': {
        'body': 'The body of a stapler is the part that holds the staples.',
        'lid': 'The lid of a stapler is the top part that can be removed to access the staples.'
    },
    'StorageFurniture': {
        'door': 'The door of a storage furniture is the part that you open to access the contents inside.',
        'drawer': 'The drawer of a storage furniture is the part that you open to access the contents inside.',
        'handle': 'The handle of a storage furniture is the part that you use to open the door or drawer.'
    },
    'Suitcase': {
        'handle': 'The handle of a suitcase is the part that you use to carry it.',
        'wheel': 'A wheel is a circular object that is used to move the suitcase.'
    },
    'Switch': {
        'switch': 'A switch is a small, usually circular object that is used to control the switch.'
    },
    'Table': {
        'door': 'The door of a table is the part that you open to access the contents inside.',
        'drawer': 'The drawer of a table is the part that you open to access the contents inside.',
        'leg': 'The leg of a table is the part that supports the table.',
        'tabletop': 'The tabletop of a table is the part that you put things on.',
        'wheel': 'A wheel is a circular object that is used to move the table.',
        'handle': 'The handle of a table is the part that you use to open the door or drawer.'
    },
    'Toaster': {
        'button': 'A button is a small, usually circular object that is used to control the toaster.',
        'slider': 'A slider is a small, usually rectangular object that is used to control the toaster.'
    },
    'Toilet': {
        'lid': 'The lid of a toilet is the top part that can be removed to access the contents inside.',
        'seat': 'The seat of a toilet is the part that you sit on.',
        'button': 'A button is a small, usually circular object that is used to control the toilet.'
    },
    'TrashCan': {
        'footpedal': 'The footpedal of a trash can is the part that you step on to open the lid.',
        'lid': 'The lid of a trash can is the top part that can be removed to access the contents inside.',
        'door': 'The door of a trash can is the part that you open to access the contents inside.'
    },
    'USB': {
        'cap': 'The cap of a USB is the top part that can be removed to access the contents inside.',
        'rotation': 'A rotation is a small, usually circular object that is used to control the USB.'
    },
    'WashingMachine': {
        'door': 'The door of a washing machine is the part that you open to put clothes inside.',
        'button': 'A button is a small, usually circular object that is used to control the washing machine.'
    },
    'Window': {
        'window': 'The window of a window is the part that you open to let air and light in.'
    }
}


objects_parts_partslip_with_parts_description_v2 = {
    "Bottle": {
        "lid": "The cap of a bottle is the top part that can be unscrewed to access the contents inside."
    },
    "Box": {
        "lid": "The cover of a box is the top part that can be lifted to access the contents inside."
    },
    "Bucket": {
        "handle": "The grip of a bucket is the part that is held to carry the bucket."
    },
    "Camera": {
        "button": "A button is a small, usually circular object that is used to capture images with the camera.",
        "lens": "The lens of a camera is the part that focuses the light onto the film or sensor."
    },
    "Cart": {
        "wheel": "A wheel is a circular object that is used to roll the cart."
    },
    "Chair": {
        "arm": "The armrest of a chair is the part that provides support for your arms.",
        "back": "The backrest of a chair is the part that supports your back.",
        "leg": "The leg of a chair is the part that contacts the ground and supports the weight.",
        "seat": "The seating surface of a chair is the part where you sit.",
        "wheel": "A wheel is a circular object that is used to move the chair."
    },
    "Clock": {
        "hand": "The pointers of a clock are the parts that move to indicate the time."
    },
    "CoffeeMachine": {
        "button": "A button is a small, usually circular object that is used to select brewing options.",
        "container": "The reservoir of a coffee machine is the part that stores water and coffee grounds.",
        "knob": "A knob is a small, usually circular object that is used to adjust settings on the coffee machine.",
        "lid": "The cover of a coffee machine is the top part that can be opened to access the interior."
    },
    "Dishwasher": {
        "door": "The hatch of a dishwasher is the part that swings open for loading and unloading dishes.",
        "handle": "The grip of a dishwasher is the part that is pulled to open the door."
    },
    "Dispenser": {
        "head": "The spout of a dispenser is the part that dispenses the contents.",
        "lid": "The top of a dispenser is the cover that can be lifted to refill or clean the dispenser."
    },
    "Display": {
        "base": "The base of a display is the part that provides stability and support.",
        "screen": "The screen of a display is the surface that visual information is presented on.",
        "support": "The stand of a display is the structure that holds and positions the screen."
    },
    "Door": {
        "frame": "The frame of a door is the structure that surrounds and supports it.",
        "door": "The leaf of a door is the movable part that opens and closes the doorway.",
        "handle": "The knob of a door is the component that is turned to operate the latch."
    },
    "Eyeglasses": {
        "body": "The frame of eyeglasses is the part that holds the lenses and connects the temples.",
        "leg": "The temple of eyeglasses is the component that extends over your ears."
    },
    "Faucet": {
        "spout": "The nozzle of a faucet is the part from which water flows.",
        "switch": "A lever is a small, usually elongated object that is used to regulate water flow."
    },
    "FoldingChair": {
        "seat": "The seating area of a folding chair is the part where you sit."
    },
    "Globe": {
        "sphere": "The sphere of a globe is the spherical component that represents the Earth's surface."
    },
    "Kettle": {
        "lid": "The cover of a kettle is the top part that can be lifted to pour water or fill it.",
        "handle": "The grip of a kettle is the part that is held while pouring.",
        "spout": "The nozzle of a kettle is the part from which water is poured."
    },
    "Keyboard": {
        "cord": "The cable of a keyboard is the part that connects it to the computer.",
        "key": "A key is a small, usually rectangular object that is pressed to input characters."
    },
    "KitchenPot": {
        "lid": "The cover of a kitchen pot is the top part that can be lifted to access the contents inside.",
        "handle": "The grip of a kitchen pot is the part that is held while pouring or moving."
    },
    "Knife": {
        "blade": "The cutting edge of a knife is the sharpened part used for slicing.",
        "handle": "The grip of a knife is the part that is held while cutting."
    },
    "Lamp": {
        "base": "The stand of a lamp is the part that provides stability and support.",
        "body": "The frame of a lamp is the part that holds the light bulb and shade.",
        "bulb": "The lightbulb of a lamp is the component that emits light.",
        "shade": "The covering of a lamp is the part that diffuses and softens light."
    },
    "Laptop": {
        "keyboard": "The input area of a laptop is the part where you type.",
        "screen": "The display of a laptop is the surface that shows visuals.",
        "shaft": "The hinge of a laptop is the part that allows the screen to tilt.",
        "touchpad": "The touch-sensitive surface of a laptop is used for cursor control.",
        "camera": "The webcam of a laptop is the component used for video capture."
    },
    "Lighter": {
        "lid": "The cover of a lighter is the top part that can be flipped open to expose the ignition mechanism.",
        "wheel": "The striker of a lighter is the part that creates the spark.",
        "button": "A trigger is a small, usually elongated object that is pressed to ignite the lighter."
    },
    "Microwave": {
        "display": "The screen of a microwave is the part that shows cooking settings and time.",
        "door": "The hatch of a microwave is the part that swings open for loading and unloading food.",
        "handle": "The grip of a microwave is the part that is pulled to open the door.",
        "button": "A keypad is an array of buttons used to input cooking parameters."
    },
    "Mouse": {
        "button": "A clicker is a small, usually circular object that is pressed to interact with the computer.",
        "cord": "The cable of a mouse is the part that connects it to the computer.",
        "wheel": "The roller of a mouse is the part that moves to scroll through content."
    },
    "Oven": {
        "door": "The hatch of an oven is the part that swings open for inserting and removing food.",
        "knob": "A dial is a small, usually circular object that is rotated to set temperature and functions."
    },
    "Pen": {
        "cap": "The cap of a pen is the top part that can be removed to reveal the writing tip.",
        "button": "A clicker is a small, usually circular object that is used to extend or retract the pen tip."
    },
    "Phone": {
        "lid": "The cover of a phone is the top part that can be flipped open to access the battery or SIM card.",
        "button": "A keypad is a set of small, usually circular objects that are pressed to input commands or numbers."
    },
    "Pliers": {
        "leg": "The grip of pliers is the part that you hold while using them.",
    },
    "Printer": {
        "button": "A button is a small, usually circular object that is used to initiate printing or scanning."
    },
    "Refrigerator": {
        "door": "The door of a refrigerator is the part that swings open to access the food compartments.",
        "handle": "The grip of a refrigerator is the part that you use to pull the door open."
    },
    "Remote": {
        "button": "A remote control typically has an array of small, usually circular objects that are pressed to operate electronic devices."
    },
    "Safe": {
        "door": "The door of a safe is the part that you open to access the contents inside.",
        "switch": "A dial is a small, usually circular object that is used to operate the locking mechanism of the safe.",
        "button": "A keypad is a set of small, usually rectangular objects that are pressed to input the combination to open the safe."
    },
    "Scissors": {
        "blade": "The cutting edge of scissors is the part that slices through material.",
        "handle": "The grip of scissors is the part that you hold.",
        "screw": "The pivot screw of scissors is the part that adjusts the tension between the blades."
    },
    "Stapler": {
        "body": "The body of a stapler is the main structure that houses the staple magazine.",
        "lid": "The cover of a stapler is the top part that can be lifted to reload staples."
    },
    "StorageFurniture": {
        "door": "The door of storage furniture is the part that swings open to access the contents inside.",
        "drawer": "The drawer of storage furniture is the part that slides out to access stored items.",
        "handle": "The grip of storage furniture is the part that you use to pull open doors or drawers."
    },
    "Suitcase": {
        "handle": "The handle of a suitcase is the part that extends and retracts for carrying.",
        "wheel": "A caster is a small, usually circular object that is attached to the suitcase for rolling."
    },
    "Switch": {
        "switch": "A switch is a small, usually rectangular object that is toggled to turn a device on or off."
    },
    "Table": {
        "door": "The hatch of a table is the part that flips open to reveal storage space underneath.",
        "drawer": "The drawer of a table is the part that slides out to access stored items.",
        "leg": "The leg of a table is the part that supports the tabletop.",
        "tabletop": "The surface of a table is the part where items are placed.",
        "wheel": "A caster is a small, usually circular object that is attached to the table for easy movement.",
        "handle": "The grip of a table is the part that you use to pull open doors or drawers."
    },
    "Toaster": {
        "button": "A button is a small, usually circular object that is used to initiate toasting or adjust settings.",
        "slider": "A lever is a small, usually rectangular object that is moved to adjust the toasting level."
    },
    "Toilet": {
        "lid": "The cover of a toilet is the top part that can be lifted for cleaning or maintenance.",
        "seat": "The seating surface of a toilet is the part that you sit on.",
        "button": "A flush handle is a small, usually elongated object that is pressed to flush the toilet."
    },
    "TrashCan": {
        "footpedal": "The foot pedal of a trash can is the part that you step on to open the lid.",
        "lid": "The cover of a trash can is the top part that can be lifted to deposit trash.",
        "door": "The door of a trash can is the part that swings open for easier disposal of larger items."
    },
    "USB": {
        "cap": "The cap of a USB is the top part that can be removed to reveal the connector.",
        "rotation": "A swivel is a small, usually circular object that is turned to retract or extend the USB connector."
    },
    "WashingMachine": {
        "door": "The hatch of a washing machine is the part that swings open for loading and unloading laundry.",
        "button": "A control panel is an array of small, usually circular objects that are pressed to select wash cycles and settings."
    },
    "Window": {
        "window": "The sash of a window is the part that slides or swings open to allow ventilation."
    }
}


objects_partnetE_description_v1 = {
    "Bottle": "A bottle is a container with a narrow neck that is used to store liquids.",
    "Box": "A box is a container with a lid that is used to store and transport items.",
    "Bucket": "A bucket is a cylindrical container with a handle that is used to carry liquids or solids.",
    "Camera": "A camera is a device that captures and stores images and videos.",
    "Cart": "A cart is a wheeled vehicle that is used to transport goods.",
    "Chair": "A chair is a piece of furniture with a raised surface that is used to sit on.",
    "Clock": "A clock is a device that shows the time.",
    "CoffeeMachine": "A coffee machine is a device that brews coffee.",
    "Dishwasher": "A dishwasher is a machine that is used to wash dishes.",
    "Dispenser": "A dispenser is a device that releases a specific amount of a substance.",
    "Display": "A display is a screen that shows images and videos.",
    "Door": "A door is a movable barrier that is used to close off an entrance or exit.",
    "Eyeglasses": "Eyeglasses are a pair of lenses set in a frame that is worn to correct vision.",
    "Faucet": "A faucet is a device that controls the flow of water.",
    "FoldingChair": "A folding chair is a portable chair that can be folded for storage.",
    "Globe": "A globe is a spherical model of the Earth.",
    "Kettle": "A kettle is a container with a spout that is used to boil water.",
    "Keyboard": "A keyboard is a set of keys that is used to input data into a computer.",
    "KitchenPot": "A kitchen pot is a container with a lid that is used to cook food.",
    "Knife": "A knife is a cutting tool with a sharp blade.",
    "Lamp": "A lamp is a device that produces light.",
    "Laptop": "A laptop is a portable computer.",
    "Lighter": "A lighter is a device that produces a flame for lighting cigarettes, candles, or other items.",
    "Microwave": "A microwave is a device that cooks food by exposing it to electromagnetic radiation.",
    "Mouse": "A mouse is a device that is used to control a computer.",
    "Oven": "An oven is a device that is used to cook food.",
    "Pen": "A pen is a writing instrument that uses ink to leave a mark on a surface.",
    "Phone": "A phone is a device that is used to make and receive calls.",
    "Pliers": "Pliers are a hand tool used to grip, turn, or bend objects.",
    "Printer": "A printer is a device that produces text or images on paper.",
    "Refrigerator": "A refrigerator is a device that is used to keep food and drinks cold.",
    "Remote": "A remote control is a device that is used to operate electronic devices from a distance.",
    "Safe": "A safe is a secure storage container that is used to protect valuables.",
    "Scissors": "Scissors are a cutting instrument with two blades.",
    "Stapler": "A stapler is a device that is used to fasten sheets of paper together.",
    "StorageFurniture": "Storage furniture is a piece of furniture that is used to store items.",
    "Suitcase": "A suitcase is a portable case that is used to carry clothes and other items.",
    "Switch": "A switch is a device that is used to control the flow of electricity.",
    "Table": "A table is a piece of furniture with a flat top that is used to put things on.",
    "Toaster": "A toaster is a device that is used to toast bread.",
    "Toilet": "A toilet is a fixture used for the disposal of bodily waste.",
    "TrashCan": "A trash can is a container that is used to hold waste.",
    "USB": "A USB is a device that is used to store and transfer data.",
    "WashingMachine": "A washing machine is a device that is used to wash clothes.",
    "Window": "A window is an opening in a wall that is used to let in light and air."
}


objects_partnetE_description_v2 = {
    "Bottle": "A bottle is a receptacle typically made of glass, plastic, or metal, with a narrow neck, used for storing and dispensing liquids.",
    "Box": "A box is a container typically made of cardboard, wood, or plastic, with an enclosed space and usually a lid, used for storage or transportation of items.",
    "Bucket": "A bucket is a cylindrical or conical container typically made of plastic or metal, with a handle attached to the side, used for carrying liquids or solids.",
    "Camera": "A camera is an optical instrument consisting of a lens system and light-sensitive sensor, used to capture and record still images or videos.",
    "Cart": "A cart is a wheeled vehicle typically made of metal or plastic, with a platform or basket, used for transporting goods or materials.",
    "Chair": "A chair is a piece of furniture typically consisting of a raised surface supported by legs, used for sitting upon.",
    "Clock": "A clock is a device that measures and displays time, typically through the use of mechanical or electronic mechanisms.",
    "CoffeeMachine": "A coffee machine is a device that brews coffee by passing hot water through ground coffee beans.",
    "Dishwasher": "A dishwasher is a machine used for cleaning dishes and utensils automatically, typically by spraying hot water and detergent.",
    "Dispenser": "A dispenser is a device that releases a specific amount of a substance, such as soap, water, or snacks, when triggered.",
    "Display": "A display is a screen or panel that presents visual information, such as images or videos, usually electronically generated.",
    "Door": "A door is a hinged or sliding panel that opens and closes to provide access or control airflow within a structure.",
    "Eyeglasses": "Eyeglasses, also known as spectacles or glasses, are optical devices consisting of lenses mounted in a frame, worn to correct or enhance vision.",
    "Faucet": "A faucet, also known as a tap, is a device for controlling the flow of a liquid, typically water, from a pipe or container.",
    "FoldingChair": "A folding chair is a portable chair that can be collapsed or folded for ease of storage or transport.",
    "Globe": "A globe is a spherical model representing the Earth's surface, typically mounted on a stand and used for reference or decoration.",
    "Kettle": "A kettle is a metal or plastic container with a handle and spout, used for boiling water.",
    "Keyboard": "A keyboard is a set of keys, usually arranged in a specific layout, used for inputting text or commands into a computer or other device.",
    "KitchenPot": "A kitchen pot is a container typically made of metal or ceramic, with a handle and lid, used for cooking food.",
    "Knife": "A knife is a cutting tool consisting of a sharp-edged blade attached to a handle, used for slicing or cutting.",
    "Lamp": "A lamp is a device for producing light, typically consisting of a bulb or tube mounted on a stand or base.",
    "Laptop": "A laptop is a portable computer designed for use on the go, typically featuring a keyboard and screen in a single unit.",
    "Lighter": "A lighter is a portable device that generates a flame, used for igniting cigarettes, candles, or other combustible materials.",
    "Microwave": "A microwave is an appliance that cooks or heats food by exposing it to electromagnetic radiation in the microwave frequency range.",
    "Mouse": "A mouse is a hand-held pointing device used to control the movement of a cursor on a computer screen.",
    "Oven": "An oven is an enclosed compartment or appliance used for heating, baking, or roasting food.",
    "Pen": "A pen is a writing instrument consisting of a slender tube containing ink, with a nib or ballpoint at one end for applying ink to a surface.",
    "Phone": "A phone is a telecommunications device used for making and receiving calls, typically consisting of a handset connected to a network.",
    "Pliers": "Pliers are hand tools consisting of two pivoted jaws used for gripping, bending, or cutting objects.",
    "Printer": "A printer is a peripheral device that produces text or graphics on paper or other media from digital input.",
    "Refrigerator": "A refrigerator is a household appliance used for preserving food and drinks at low temperatures.",
    "Remote": "A remote control is a handheld device or interface used to operate electronic devices wirelessly from a distance.",
    "Safe": "A safe is a secure storage container typically made of metal, used for protecting valuables or important documents from theft or damage.",
    "Scissors": "Scissors are cutting instruments consisting of two blades pivoted together, used for cutting various materials.",
    "Stapler": "A stapler is a device for fastening sheets of paper together by driving a metal staple through them.",
    "StorageFurniture": "Storage furniture is furniture designed to store items, often featuring drawers, shelves, or compartments.",
    "Suitcase": "A suitcase is a rectangular container with a handle, typically made of leather, fabric, or plastic, used for carrying clothes and personal belongings while traveling.",
    "Switch": "A switch is a mechanical or electronic device used to control the flow of electricity within a circuit.",
    "Table": "A table is a piece of furniture with a flat horizontal surface, typically supported by one or more legs, used for various activities such as dining, working, or playing games.",
    "Toaster": "A toaster is a small kitchen appliance used for browning slices of bread by exposing them to radiant heat.",
    "Toilet": "A toilet is a fixture used for the disposal of bodily waste, typically consisting of a bowl connected to a drain and flushing mechanism.",
    "TrashCan": "A trash can, also known as a garbage bin or waste basket, is a container used for the disposal and temporary storage of waste materials.",
    "USB": "A USB, short for Universal Serial Bus, is a standard interface used for connecting various devices to a computer or other electronic device for data transfer or charging.",
    "WashingMachine": "A washing machine is an appliance used for washing clothes, typically consisting of a drum that agitates water and detergent to clean the garments.",
    "Window": "A window is an opening in a wall or vehicle that allows light and air to enter and provides a view to the outside."
}