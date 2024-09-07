
# Model
feat_dims = {'ViT-B/16':512, 'ViT-B/32':512, 'RN50':1024, 'RN101':512}

# Data
cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
            'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
            'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}

# Part
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 2, 2, 6, 2, 3, 3, 3, 3]

# Start part id for each category
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

id2cat = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
        'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']

cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['leg or wire','lampshade'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
id2part2cat = [['body', 'airplane'], ['wing', 'airplane'], ['tail', 'airplane'], ['engine or frame', 'airplane'], ['handle', 'bag'], ['body', 'bag'], 
            ['panels or crown', 'cap'], ['visor or peak', 'cap'],
            ['roof', 'car'], ['hood', 'car'], ['wheel or tire',  'car'], ['body', 'car'],
            ['backrest or back', 'chair'], ['seat', 'chair'], ['leg or support', 'chair'], ['armrest', 'chair'], 
            ['earcup', 'earphone'], ['headband', 'earphone'], ['data wire',  'earphone'], 
            ['head or tuners', 'guitar'], ['neck', 'guitar'], ['body', 'guitar'], ['blade', 'knife'], ['handle', 'knife'], 
            ['support or tube of wire', 'lamp'], ['lampshade', 'lamp'], ['canopy', 'lamp'], ['support or tube of wire', 'lamp'], 
            ['keyboard', 'laptop'], ['screen or monitor', 'laptop'], ['gas tank', 'motorbike'], ['seat', 'motorbike'], ['wheel', 'motorbike'], 
            ['handles or handlebars', 'motorbike'], ['light', 'motorbike'], ['engine or frame', 'motorbike'], ['handle', 'mug'], ['cup or body', 'mug'], 
            ['barrel', 'pistol'], ['handle', 'pistol'], ['trigger and guard', 'pistol'], ['body', 'rocket'], ['fin', 'rocket'], ['nose cone', 'rocket'], 
            ['wheel', 'skateboard'], ['deck',  'skateboard'], ['belt for foot', 'skateboard'], 
            ['desktop', 'table'], ['leg or support', 'table'], ['drawer''table']]

# Best prompts provided by the authors
best_prompt = {
    'airplane': [' The nose is the front part of the aircraft that houses the cockpit.', 'There is no wing part of an airplane in this grayscale map.', ' The back end of an airplane, with its two engines and two methods of steering.', 'This sentence is describing a partial airplane that is being shown in a depth map.'],
    'bag': ['The bag has a black strap that goes over the shoulder.', ' Ready for Literally AnythingThis bag is perfect for carrying all of your essentials with you on the go.'],
    'cap': ['Assuming you are talking about a baseball cap, the crown is typically the highest point of the hat, and the panels are the pieces of fabric that make up the sides and back of the hat.', 'This sentence is about the peak value of the grayscale depth map.'],
    'car': ['the car has a metal roof that is slanted down towards the back.', 'The hood of a car is an important part of the vehicle.', 'I need new tires for my car.', 'The engine is the heart of a car.'],
    'chair': ['Back of the chair depth map.', ' A closeup of the seat pad of a chair.', 'A leg part of a chair 3D model can look like a cylinder, a square, or a rectangle.', 'In a chair depth map, an armrest would appear as a horizontal line at the appropriate depth.'],
    'earphone': ['This is the earcup of a earphone in a three-dimensional map.', 'This is the headband part of an earphone in a 3D map.', 'A typical earphone has a wire that consists of four parts: the inner conductor, the dielectric insulation, the outer conductor, and the jacket.'],
    'guitar': ['This sentence is saying that the head or tuning pegs are the only part of the guitar that is shown in the depth map.', 'It is a representation of a gray 3D guitar model.', 'The body part of a guitar can be identified in this grayscale map by its shape.'],
    'knife': ['A depth map of a knife typically shows the blade as a thin, straight line, while the handle may be thicker and more curved.', 'A handle part of a knife 3D model might look like a cylindrical piece with a hole in the center for the blade to fit into.'],
    'lamp': ['There is no one-size-fits-all answer to this question, as the best way to segment the leg or wire part of a lamp in a depth map may vary depending on the.', 'Since this is a depth map, you can segment the lampshade by finding the points in the depth map that correspond to the lampshade.'],
    'laptop': ['The keyboard feature of a laptop 3D model is that it is a separate object that can be moved around and positioned as desired.', 'Laptop computer with screen open, viewed from above.'],
    'motorbike': ["The gas tank's motorbike would appear as a dark object in a grayscale depth map.", 'There is no easy answer for this question.', 'This sentence is describing a wheel on a motorcycle in a photograph.', 'There is no definitive answer to this question as it depends on the specific depth map and the desired outcome.', 'There is no definitive answer to this question since it will vary depending on the desired outcome.', 'The engine is the "heart" of the bike.'],
    'mug': ['This sentence is describing a depth map, which is a tool used in computer vision to create a representation of the surfaces of a scene from a set of digital images.', 'Only the bottom part of this mug is recognized.'],
    'pistol': ['This is the part of the pistol depth map that shows the barrel.', 'The part of the pistol that you would hold in your hand is the grip.', 'Thesynonym of this sentence is: The trigger and guard of a gun.'],
    'rocket': ['ROCKET BODYThis is the body of a rocket.', 'A fin is typically a thin, flat surface that is attached to the back end of a rocket.', 'A nose cone on a rocket 3D model typically looks like a cone or pyramid shape.'],
    'skateboard': ['The depth map of the wheel on a skateboard is important.', 'Caption: The deck of a skateboard, viewed from the top.', 'This sentence is describing a strap or belt that goes around the foot of a skateboard.'],
    'table': ['A depth map of a table, showing the desktop at the top and the underside of the table at the bottom.', 'The table is a rectangle with a light gray color.', 'The table is a rectangle with a light gray color.'],
}

# Best weights provided by the authors  
best_vweight = {
    'airplane': [0.75, 0.75, 0.25, 0.25, 0.25, 0.50, 1.00, 0.25, 0.25, 0.25],
    'bag': [0.75, 0.75, 0.25, 0.75, 1.00, 0.25, 1.00, 0.50, 0.25, 0.25],
    'cap': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'car': [0.75, 0.75, 0.25, 0.25, 0.25, 0.75, 0.25, 0.75, 1.00, 0.25],
    'chair': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'earphone': [0.75, 0.75, 0.25, 0.25, 0.25, 0.25, 0.75, 0.50, 0.25, 0.50],
    'guitar': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'knife': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'lamp': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'laptop': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'motorbike': [0.75, 0.75, 0.25, 0.25, 0.50, 0.75, 0.25, 0.75, 1.00, 0.25],
    'mug': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'pistol': [0.75, 0.75, 0.25, 0.25, 0.25, 1.00, 0.25, 1.00, 0.75, 0.25],
    'rocket': [0.75, 0.75, 0.25, 0.25, 0.50, 1.00, 0.25, 0.50, 0.25, 0.75],
    'skateboard': [0.75, 0.75, 0.25, 0.50, 0.25, 1.00, 0.50, 0.25, 0.75, 1.00],
    'table': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

# Object descriptions
best_obj_descriptions = {
    'airplane': 'The airplane is a vehicle that can fly through the air. It has wings and engines.',
    'bag': 'A bag is a flexible container with a single opening.',
    'cap': 'A cap is a form of headgear. Caps have crowns that fit very close to the head and have no brim or only a visor.',
    'car': 'A car is a road vehicle used to carry passengers. It is also called an automobile.',
    'chair': 'A chair is a piece of furniture with a raised surface supported by legs.',
    'earphone': 'An earphone is a small loudspeaker that is worn over or inside the ear.',
    'guitar': 'A guitar is a musical instrument that usually has six strings.',
    'knife': 'A knife is a tool with a cutting edge or blade, often attached to a handle or hilt.',
    'lamp': 'A lamp is a device that produces light by the flow of electrical current.',
    'laptop': 'A laptop is a small, portable personal computer with a "clamshell" form factor.',
    'motorbike': 'A motorbike is a two-wheeled vehicle that is powered by a motor.',
    'mug': 'A mug is a type of cup typically used for drinking hot beverages, such as coffee, hot chocolate, soup, or tea.',
    'pistol': 'A pistol is a type of handgun.',
    'rocket': 'A rocket is a vehicle that is propelled by the ejection of exhaust gases from a rocket engine.',
    'skateboard': 'A skateboard is a type of sports equipment used for skateboarding.',
    'table': 'A table is an item of furniture with a flat top and one or more legs.'
}

# best prompts generated by GPT-3
best_prompt_gpt3 = {
  "airplane": [
    "The cockpit is housed in the front part of the aircraft, known as the nose.",
    "This grayscale map does not display the wing section of the airplane.",
    "Located at the rear end of the airplane are two engines and two steering mechanisms.",
    "The depth map reveals only a portion of the airplane, likely showing its rear section."
  ],
  "bag": [
    "The bag features a black shoulder strap for easy carrying.",
    "This versatile bag is ideal for carrying essential items on the go."
  ],
  "cap": [
    "In reference to a baseball cap, the highest point is called the crown, while the sides and back are made up of panels.",
    "This statement pertains to the maximum value depicted in the grayscale depth map."
  ],
  "car": [
    "The car's roof is slanted downward towards the rear and made of metal.",
    "A crucial component of the vehicle, the car's hood, is highlighted.",
    "Replacement tires are needed for the car.",
    "The engine serves as the vital center of the car."
  ],
  "chair": [
    "The depth map illustrates the back of the chair.",
    "An up-close depiction of the chair's seat pad.",
    "The leg of a chair 3D model may take the form of a cylinder, square, or rectangle.",
    "In a chair's depth map, an armrest would be represented by a horizontal line at a specific depth."
  ],
  "earphone": [
    "This three-dimensional map showcases the earcup of an earphone.",
    "Displayed in the 3D map is the headband segment of an earphone.",
    "A standard earphone wire consists of four parts: the inner conductor, dielectric insulation, outer conductor, and jacket."
  ],
  "guitar": [
    "The depth map exclusively features the head or tuning pegs of the guitar.",
    "A grayscale 3D model representation is presented.",
    "The body of the guitar can be identified by its shape in this grayscale map."
  ],
  "knife": [
    "In a typical knife depth map, the blade appears as a thin, straight line, while the handle may have a thicker, curved design.",
    "The handle of a knife 3D model might resemble a cylindrical piece with a central hole for the blade."
  ],
  "lamp": [
    "Segmenting the leg or wire section of a lamp in a depth map may vary depending on the specific context.",
    "Identifying the lampshade involves locating corresponding points in the depth map."
  ],
  "laptop": [
    "The keyboard is a separate movable object in the laptop's 3D model.",
    "An overhead view of a laptop computer with the screen open."
  ],
  "motorbike": [
    "The gas tank of a motorbike would be represented as a dark object in a grayscale depth map.",
    "This question lacks a straightforward answer.",
    "This statement describes a wheel on a motorcycle depicted in a photograph.",
    "The answer to this question varies based on the specific depth map and desired outcome.",
    "Responses may differ depending on the desired outcome.",
    "The engine is the essential component of the bike."
  ],
  "mug": [
    "Describing a tool used in computer vision, known as a depth map, to create scene surface representations from digital images.",
    "Only the bottom portion of this mug is identifiable."
  ],
  "pistol": [
    "The pistol depth map highlights the barrel section.",
    "The grip is the part of the pistol held in the hand.",
    "A synonymous statement would be: The trigger and guard of a gun."
  ],
  "rocket": [
    "This depicts the body of a rocket.",
    "A fin, typically a thin, flat surface, is attached to the rear of a rocket.",
    "A nose cone on a rocket 3D model typically exhibits a cone or pyramid shape."
  ],
  "skateboard": [
    "The depth map emphasizes the skateboard's wheel.",
    "Caption: Top view of the skateboard deck.",
    "Describing a strap or belt that secures the foot on a skateboard."
  ],
  "table": [
    "A depth map of a table displays the top surface as well as the underside.",
    "The table is depicted as a rectangular shape with a light gray hue.",
    "The table is depicted as a rectangular shape with a light gray hue."
  ]
}
