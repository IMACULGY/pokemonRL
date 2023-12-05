from enum import Enum

moves = {
    "Absorb": ["Grass", 20, 100],
    "Acid": ["Poison", 40, 100],
    "Acid Armor": ["Poison", 0, 0],
    "Agility": ["Psychic", 0, 0],
    "Amnesia": ["Psychic", 0, 0],
    "Aurora Beam": ["Ice", 65, 100],
    "Barrage": ["Normal", 15, 85],
    "Barrier": ["Psychic", 0, 0],
    "Bide": ["Normal", "Var Dmg", 100],
    "Bind": ["Normal", 15, 75],
    "Bite": ["Normal", 60, 100],
    "Blizzard": ["Ice", 120, 90],
    "Body Slam": ["Normal", 85, 100],
    "Bone Club": ["Ground", 65, 85],
    "Bonemerang": ["Ground", 50, 90],
    "Bubble": ["Water", 20, 100],
    "Bubblebeam": ["Water", 65, 100],
    "Clamp": ["Water", 35, 75],
    "Comet Punch": ["Normal", 18, 85],
    "Confuse Ray": ["Ghost", 0, 100],
    "Confusion": ["Psychic", 50, 100],
    "Constrict": ["Normal", 10, 100],
    "Conversion": ["Normal", 0, 0],
    "Counter": ["Fighting", "Var Dmg", 100],
    "Crabhammer": ["Water", 90, 85],
    "Cut": ["Normal", 50, 95],
    "Defense Curl": ["Normal", 0, 0],
    "Dig": ["Ground", 100, 100],
    "Disable": ["Normal", 0, 55],
    "Dizzy Punch": ["Normal", 70, 100],
    "Double Kick": ["Fighting", 30, 100],
    "Double Team": ["Normal", 0, 0],
    "Double Edge": ["Normal", 100, 100],
    "Doubleslap": ["Normal", 15, 85],
    "Dragon Rage": ["Dragon", "Set", 100],
    "Dream Eater": ["Psychic", 100, 100],
    "Drill Peck": ["Flying", 80, 100],
    "Earthquake": ["Ground", 100, 100],
    "Egg Bomb": ["Normal", 100, 75],
    "Ember": ["Fire", 40, 100],
    "Explosion": ["Normal", 170, 100],
    "Fire Blast": ["Fire", 120, 85],
    "Fire Punch": ["Fire", 75, 100],
    "Fire Spin": ["Fire", 15, 70],
    "Fissure": ["Ground", "KO", 30],
    "Flamethrower": ["Fire", 95, 100],
    "Flash": ["Normal", 0, 70],
    "Fly": ["Flying", 70, 95],
    "Focus Energy": ["Normal", 0, 0],
    "Fury Attack": ["Normal", 15, 85],
    "Fury Swipes": ["Normal", 10, 80],
    "Glare": ["Normal", 0, 75],
    "Growl": ["Normal", 0, 100],
    "Growth": ["Normal", 0, 0],
    "Guillotine": ["Normal", "KO", 30],
    "Gust": ["Normal", 40, 100],
    "Harden": ["Normal", 0, 0],
    "Haze": ["Ice", 0, 0],
    "Headbutt": ["Normal", 70, 100],
    "Hi Jump Kick": ["Fighting", 85, 90],
    "Horn Attack": ["Normal", 65, 100],
    "Horn Drill": ["Normal", "KO", 30],
    "Hydro Pump": ["Water", 120, 80],
    "Hyper Beam": ["Normal", 150, 90],
    "Hyper Fang": ["Normal", 80, 90],
    "Hypnosis": ["Psychic", 0, 60],
    "Ice Beam": ["Ice", 95, 100],
    "Ice Punch": ["Ice", 75, 100],
    "Jump Kick": ["Fighting", 70, 95],
    "Karate Chop": ["Normal", 55, 100],
    "Kinesis": ["Psychic", 0, 80],
    "Leech Life": ["Bug", 20, 100],
    "Leech Seed": ["Grass", 0, 90],
    "Leer": ["Normal", 0, 100],
    "Lick": ["Ghost", 20, 100],
    "Light Screen": ["Psychic", 0, 0],
    "Lovely Kiss": ["Normal", 0, 75],
    "Low Kick": ["Fighting", 50, 90],
    "Meditate": ["Psychic", 0, 0],
    "Mega Drain": ["Grass", 40, 100],
    "Mega Kick": ["Normal", 120, 75],
    "Mega Punch": ["Normal", 80, 85],
    "Metronome": ["Normal", 0, 0],
    "Mimic": ["Normal", 0, 100],
    "Minimize": ["Normal", 0, 0],
    "Mirror Move": ["Flying", "Copy", 0],
    "Mist": ["Ice", 0, 0],
    "Night Shade": ["Ghost", "Var Dmg", 100],
    "Pay Day": ["Normal", 40, 100],
    "Peck": ["Flying", 35, 100],
    "Petal Dance": ["Grass", 70, 100],
    "Pin Missile": ["Bug", 14, 85],
    "Poison Gas": ["Poison", 0, 55],
    "Poison Sting": ["Poison", 15, 100],
    "Poisonpowder": ["Poison", 0, 75],
    "Pound": ["Normal", 40, 100],
    "Psybeam": ["Psychic", 65, 100],
    "Psychic": ["Psychic", 90, 100],
    "Psywave": ["Psychic", "Var Dmg", 80],
    "Quick Attack": ["Normal", 40, 100],
    "Rage": ["Normal", 20, 100],
    "Razor Leaf": ["Grass", 55, 95],
    "Razor Wind": ["Normal", 80, 75],
    "Recover": ["Normal", 0, 0],
    "Reflect": ["Psychic", 0, 0],
    "Rest": ["Psychic", 0, 0],
    "Roar": ["Normal", 0, 100],
    "Rock Slide": ["Rock", 75, 90],
    "Rock Throw": ["Rock", 50, 65],
    "Rolling Kick": ["Fighting", 60, 85],
    "Sand Attack": ["Ground", 0, 100],
    "Scratch": ["Normal", 40, 100],
    "Screech": ["Normal", 0, 85],
    "Seismic Toss": ["Fighting", "Var Dmg", 100],
    "Selfdestruct": ["Normal", 130, 100],
    "Sharpen": ["Normal", 0, 0],
    "Sing": ["Normal", 0, 55],
    "Skull Bash": ["Normal", 100, 100],
    "Sky Attack": ["Flying", 140, 90],
    "Slam": ["Normal", 80, 75],
    "Slash": ["Normal", 70, 100],
    "Sleep Powder": ["Grass", 0, 75],
    "Sludge": ["Poison", 65, 100],
    "Smog": ["Poison", 20, 100],
    "Smokescreen": ["Normal", 0, 100],
    "Softboiled": ["Normal", 0, 0],
    "Solarbeam": ["Grass", 120, 100],
    "Sonicboom": ["Normal", "Set", 90],
    "Spike Cannon": ["Normal", 20, 100],
    "Splash": ["Normal", 0, 0],
    "Spore": ["Grass", 0, 100],
    "Stomp": ["Normal", 65, 100],
    "Strength": ["Normal", 80, 100],
    "String Shot": ["Bug", 0, 95],
    "Struggle": ["Normal", 50, 100],
    "Stun Spore": ["Grass", 0, 75],
    "Submission": ["Fighting", 80, 80],
    "Substitute": ["Normal", 0, 0],
    "Super Fang": ["Normal", "Var Dmg", 90],
    "Supersonic": ["Normal", 0, 55],
    "Surf": ["Water", 95, 100],
    "Swift": ["Normal", 60, 100],
    "Swords Dance": ["Normal", 0, 0],
    "Tackle": ["Normal", 35, 95],
    "Tail Whip": ["Normal", 0, 100],
    "Take Down": ["Normal", 90, 85],
    "Teleport": ["Psychic", 0, 0],
    "Thrash": ["Normal", 90, 100],
    "Thunder": ["Electric", 120, 70],
    "Thunder Wave": ["Electric", 0, 100],
    "Thunderbolt": ["Electric", 95, 100],
    "Thunderpunch": ["Electric", 75, 100],
    "Thundershock": ["Electric", 40, 100],
    "Toxic": ["Poison", 0, 85],
    "Transform": ["Normal", 0, 0],
    "Tri Attack": ["Normal", 80, 100],
    "Twineedle": ["Bug", 25, 100],
    "Vicegrip": ["Normal", 55, 100],
    "Vine Whip": ["Grass", 35, 100],
    "Water Gun": ["Water", 40, 100],
    "Waterfall": ["Water", 80, 100],
    "Whirlwind": ["Normal", 0, 100],
    "Wing Attack": ["Normal", 35, 100],
    "Withdraw": ["Normal", 0, 100],
    "Wrap": ["Normal", 15, 85]
}

class MoveId(Enum):
    POUND = 0x01
    KARATE_CHOP = 0x02
    DOUBLESLAP = 0x03
    COMET_PUNCH = 0x04
    MEGA_PUNCH = 0x05
    PAY_DAY = 0x06
    FIRE_PUNCH = 0x07
    ICE_PUNCH = 0x08
    THUNDERPUNCH = 0x09
    SCRATCH = 0x0a
    VICEGRIP = 0x0b
    GUILLOTINE = 0x0c
    RAZOR_WIND = 0x0d
    SWORDS_DANCE = 0x0e
    CUT = 0x0f
    GUST = 0x10
    WING_ATTACK = 0x11
    WHIRLWIND = 0x12
    FLY = 0x13
    BIND = 0x14
    SLAM = 0x15
    VINE_WHIP = 0x16
    STOMP = 0x17
    DOUBLE_KICK = 0x18
    MEGA_KICK = 0x19
    JUMP_KICK = 0x1a
    ROLLING_KICK = 0x1b
    SAND_ATTACK = 0x1c
    HEADBUTT = 0x1d
    HORN_ATTACK = 0x1e
    FURY_ATTACK = 0x1f
    HORN_DRILL = 0x20
    TACKLE = 0x21
    BODY_SLAM = 0x22
    WRAP = 0x23
    TAKE_DOWN = 0x24
    THRASH = 0x25
    DOUBLE_EDGE = 0x26
    TAIL_WHIP = 0x27
    POISON_STING = 0x28
    TWINEEDLE = 0x29
    PIN_MISSILE = 0x2a
    LEER = 0x2b
    BITE = 0x2c
    GROWL = 0x2d
    ROAR = 0x2e
    SING = 0x2f
    SUPERSONIC = 0x30
    SONICBOOM = 0x31
    DISABLE = 0x32
    ACID = 0x33
    EMBER = 0x34
    FLAMETHROWER = 0x35
    MIST = 0x36
    WATER_GUN = 0x37
    HYDRO_PUMP = 0x38
    SURF = 0x39
    ICE_BEAM = 0x3a
    BLIZZARD = 0x3b
    PSYBEAM = 0x3c
    BUBBLEBEAM = 0x3d
    AURORA_BEAM = 0x3e
    HYPER_BEAM = 0x3f
    PECK = 0x40
    DRILL_PECK = 0x41
    SUBMISSION = 0x42
    LOW_KICK = 0x43
    COUNTER = 0x44
    SEISMIC_TOSS = 0x45
    STRENGTH = 0x46
    ABSORB = 0x47
    MEGA_DRAIN = 0x48
    LEECH_SEED = 0x49
    GROWTH = 0x4a
    RAZOR_LEAF = 0x4b
    SOLARBEAM = 0x4c
    POISONPOWDER = 0x4d
    STUN_SPORE = 0x4e
    SLEEP_POWDER = 0x4f
    PETAL_DANCE = 0x50
    STRING_SHOT = 0x51
    DRAGON_RAGE = 0x52
    FIRE_SPIN = 0x53
    THUNDERSHOCK = 0x54
    THUNDERBOLT = 0x55
    THUNDER_WAVE = 0x56
    THUNDER = 0x57
    ROCK_THROW = 0x58
    EARTHQUAKE = 0x59
    FISSURE = 0x5a
    DIG = 0x5b
    TOXIC = 0x5c
    CONFUSION = 0x5d
    PSYCHIC = 0x5e
    HYPNOSIS = 0x5f
    MEDITATE = 0x60
    AGILITY = 0x61
    QUICK_ATTACK = 0x62
    RAGE = 0x63
    TELEPORT = 0x64
    NIGHT_SHADE = 0x65
    MIMIC = 0x66
    SCREECH = 0x67
    DOUBLE_TEAM = 0x68
    RECOVER = 0x69
    HARDEN = 0x6a
    MINIMIZE = 0x6b
    SMOKESCREEN = 0x6c
    CONFUSE_RAY = 0x6d
    WITHDRAW = 0x6e
    DEFENSE_CURL = 0x6f
    BARRIER = 0x70
    LIGHT_SCREEN = 0x71
    HAZE = 0x72
    REFLECT = 0x73
    FOCUS_ENERGY = 0x74
    BIDE = 0x75
    METRONOME = 0x76
    MIRROR_MOVE = 0x77
    SELFDESTRUCT = 0x78
    EGG_BOMB = 0x79
    LICK = 0x7a
    SMOG = 0x7b
    SLUDGE = 0x7c
    BONE_CLUB = 0x7d
    FIRE_BLAST = 0x7e
    WATERFALL = 0x7f
    CLAMP = 0x80
    SWIFT = 0x81
    SKULL_BASH = 0x82
    SPIKE_CANNON = 0x83
    CONSTRICT = 0x84
    AMNESIA = 0x85
    KINESIS = 0x86
    SOFTBOILED = 0x87
    HI_JUMP_KICK = 0x88
    GLARE = 0x89
    DREAM_EATER = 0x8a
    POISON_GAS = 0x8b
    BARRAGE = 0x8c
    LEECH_LIFE = 0x8d
    LOVELY_KISS = 0x8e
    SKY_ATTACK = 0x8f
    TRANSFORM = 0x90
    BUBBLE = 0x91
    DIZZY_PUNCH = 0x92
    SPORE = 0x93
    FLASH = 0x94
    PSYWAVE = 0x95
    SPLASH = 0x96
    ACID_ARMOR = 0x97
    CRABHAMMER = 0x98
    EXPLOSION = 0x99
    FURY_SWIPES = 0x9a
    BONEMERANG = 0x9b
    REST = 0x9c
    ROCK_SLIDE = 0x9d
    HYPER_FANG = 0x9e
    SHARPEN = 0x9f
    CONVERSION = 0xa0
    TRI_ATTACK = 0xa1
    SUPER_FANG = 0xa2
    SLASH = 0xa3
    SUBSTITUTE = 0xa4

# List of every move name, offset by 1 from its hex value in the game
# Struggle is the only move not included.
moves_list = [move.name.replace("_", " ").title() for move in MoveId]

class PokemonId(Enum):
    RHYDON = 0x01
    KANGASKHAN = 0x02
    NIDORAN_M = 0x03
    CLEFAIRY = 0x04
    SPEAROW = 0x05
    VOLTORB = 0x06
    NIDOKING = 0x07
    SLOWBRO = 0x08
    IVYSAUR = 0x09
    EXEGGUTOR = 0x0A
    LICKITUNG = 0x0B
    EXEGGCUTE = 0x0C
    GRIMER = 0x0D
    GENGAR = 0x0E
    NIDORAN_F = 0x0F
    NIDOQUEEN = 0x10
    CUBONE = 0x11
    RHYHORN = 0x12
    LAPRAS = 0x13
    ARCANINE = 0x14
    MEW = 0x15
    GYARADOS = 0x16
    SHELLDER = 0x17
    TENTACOOL = 0x18
    GASTLY = 0x19
    SCYTHER = 0x1A
    STARYU = 0x1B
    BLASTOISE = 0x1C
    PINSIR = 0x1D
    TANGELA = 0x1E
    MISSINGNO_1F = 0x1F
    MISSINGNO_20 = 0x20
    GROWLITHE = 0x21
    ONIX = 0x22
    FEAROW = 0x23
    PIDGEY = 0x24
    SLOWPOKE = 0x25
    KADABRA = 0x26
    GRAVELER = 0x27
    CHANSEY = 0x28
    MACHOKE = 0x29
    MR_MIME = 0x2A
    HITMONLEE = 0x2B
    HITMONCHAN = 0x2C
    ARBOK = 0x2D
    PARASECT = 0x2E
    PSYDUCK = 0x2F
    DROWZEE = 0x30
    GOLEM = 0x31
    MISSINGNO_32 = 0x32
    MAGMAR = 0x33
    MISSINGNO_34 = 0x34
    ELECTABUZZ = 0x35
    MAGNETON = 0x36
    KOFFING = 0x37
    MISSINGNO_38 = 0x38
    MANKEY = 0x39
    SEEL = 0x3A
    DIGLETT = 0x3B
    TAUROS = 0x3C
    MISSINGNO_3D = 0x3D
    MISSINGNO_3E = 0x3E
    MISSINGNO_3F = 0x3F
    FARFETCHD = 0x40
    VENONAT = 0x41
    DRAGONITE = 0x42
    MISSINGNO_43 = 0x43
    MISSINGNO_44 = 0x44
    MISSINGNO_45 = 0x45
    DODUO = 0x46
    POLIWAG = 0x47
    JYNX = 0x48
    MOLTRES = 0x49
    ARTICUNO = 0x4A
    ZAPDOS = 0x4B
    DITTO = 0x4C
    MEOWTH = 0x4D
    KRABBY = 0x4E
    MISSINGNO_4F = 0x4F
    MISSINGNO_50 = 0x50
    MISSINGNO_51 = 0x51
    VULPIX = 0x52
    NINETALES = 0x53
    PIKACHU = 0x54
    RAICHU = 0x55
    MISSINGNO_56 = 0x56
    MISSINGNO_57 = 0x57
    DRATINI = 0x58
    DRAGONAIR = 0x59
    KABUTO = 0x5A
    KABUTOPS = 0x5B
    HORSEA = 0x5C
    SEADRA = 0x5D
    MISSINGNO_5E = 0x5E
    MISSINGNO_5F = 0x5F
    SANDSHREW = 0x60
    SANDSLASH = 0x61
    OMANYTE = 0x62
    OMASTAR = 0x63
    JIGGLYPUFF = 0x64
    WIGGLYTUFF = 0x65
    EEVEE = 0x66
    FLAREON = 0x67
    JOLTEON = 0x68
    VAPOREON = 0x69
    MACHOP = 0x6A
    ZUBAT = 0x6B
    EKANS = 0x6C
    PARAS = 0x6D
    POLIWHIRL = 0x6E
    POLIWRATH = 0x6F
    WEEDLE = 0x70
    KAKUNA = 0x71
    BEEDRILL = 0x72
    MISSINGNO_73 = 0x73
    DODRIO = 0x74
    PRIMEAPE = 0x75
    DUGTRIO = 0x76
    VENOMOTH = 0x77
    DEWGONG = 0x78
    MISSINGNO_79 = 0x79
    MISSINGNO_7A = 0x7A
    CATERPIE = 0x7B
    METAPOD = 0x7C
    BUTTERFREE = 0x7D
    MACHAMP = 0x7E
    MISSINGNO_7F = 0x7F
    GOLDUCK = 0x80
    HYPNO = 0x81
    GOLBAT = 0x82
    MEWTWO = 0x83
    SNORLAX = 0x84
    MAGIKARP = 0x85
    MISSINGNO_86 = 0x86
    MISSINGNO_87 = 0x87
    MUK = 0x88
    MISSINGNO_89 = 0x89
    KINGLER = 0x8A
    CLOYSTER = 0x8B
    MISSINGNO_8C = 0x8C
    ELECTRODE = 0x8D
    CLEFABLE = 0x8E
    WEEZING = 0x8F
    PERSIAN = 0x90
    MAROWAK = 0x91
    MISSINGNO_92 = 0x92
    HAUNTER = 0x93
    ABRA = 0x94
    ALAKAZAM = 0x95
    PIDGEOTTO = 0x96
    PIDGEOT = 0x97
    STARMIE = 0x98
    BULBASAUR = 0x99
    VENUSAUR = 0x9A
    TENTACRUEL = 0x9B
    MISSINGNO_9C = 0x9C
    GOLDEEN = 0x9D
    SEAKING = 0x9E
    MISSINGNO_9F = 0x9F
    MISSINGNO_A0 = 0xA0
    MISSINGNO_A1 = 0xA1
    MISSINGNO_A2 = 0xA2
    PONYTA = 0xA3
    RAPIDASH = 0xA4
    RATTATA = 0xA5
    RATICATE = 0xA6
    NIDORINO = 0xA7
    NIDORINA = 0xA8
    GEODUDE = 0xA9
    PORYGON = 0xAA
    AERODACTYL = 0xAB
    MISSINGNO_AC = 0xAC
    MAGNEMITE = 0xAD
    MISSINGNO_AE = 0xAE
    MISSINGNO_AF = 0xAF
    CHARMANDER = 0xB0
    SQUIRTLE = 0xB1
    CHARMELEON = 0xB2
    WARTORTLE = 0xB3
    CHARIZARD = 0xB4
    MISSINGNO_B5 = 0xB5
    FOSSIL_KABUTOPS = 0xB6
    FOSSIL_AERODACTYL = 0xB7
    MON_GHOST = 0xB8
    ODDISH = 0xB9
    GLOOM = 0xBA
    VILEPLUME = 0xBB
    BELLSPROUT = 0xBC
    WEEPINBELL = 0xBD
    VICTREEBEL = 0xBE

pokemon_list = [pokemon.name.replace("_", " ").title() for pokemon in PokemonId]

types = {
    "Normal": 1,
    "Fighting": 2,
    "Flying": 3,
    "Poison": 4,
    "Ground": 5,
    "Rock": 6,
    "Bug": 7,
    "Ghost": 8,
    "Fire": 9,
    "Water": 10,
    "Grass": 11,
    "Electric": 12,
    "Psychic": 13,
    "Ice": 14,
    "Dragon": 15
}