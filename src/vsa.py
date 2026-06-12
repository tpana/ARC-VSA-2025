# Isaac Joffe, 2025
# Defines VSA setup for representation

# VSA-related dependencies
import numpy as np
import nengo_spa as spa
import sspspace
# Machine-learning-related dependencies
from sklearn.mixture import GaussianMixture
# General helpers
from PIL import ImageColor
import random

from scipy.optimize import *

# Control how random behaviour is set
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
RNG = np.random.RandomState(SEED)


# General VSA setup
# -----------------------------------------------------------------------------
# ARC hyperparameters
N_COLOURS = 10        # Black, blue, red, green, yellow, gray, pink, orange, cyan, maroon
MAX_GRID_SIZE = 30    # At most 30x30 size grids

# VSA hyperparameters
N_DIMENSIONS = int(8192/2)          # How many dimensions to use in embedding
LENGTH_SCALE = 0.25           # How 'blurred' to make all representations

# System hyperparameters
N_FEATURES = 6        # Colour, centre, shape, number, height, width
MAX_N_OBJECTS = 10    # At most 10 input and 10 output objects

# Generate an SP for each object tag
object_tags = [
    "OBJECT_0",
    "OBJECT_1",
    "OBJECT_2",
    "OBJECT_3",
    "OBJECT_4",
    "OBJECT_5",
    "OBJECT_6",
    "OBJECT_7",
    "OBJECT_8",
    "OBJECT_9",
]

# Generate an SP for each feature slot
feature_tags = [
    "COLOUR",
    "CENTRE",
    "SHAPE",
    "NUMBER",
    "HEIGHT",
    "WIDTH",
]

# Generate an SP for each ARC colour
colour_tags = [
    "BLACK",
    "BLUE",
    "RED",
    "GREEN",
    "YELLOW",
    "GREY",
    "PINK",
    "ORANGE",
    "CYAN",
    "MAROON",
]

# Generate an SP for each possible number
number_tags = [
    "ZERO.unitary()",
    "ONE.unitary()",
    "TWO = ONE * ONE",
    "THREE = TWO * ONE",
    "FOUR = THREE * ONE",
    "FIVE = FOUR * ONE",
    "SIX = FIVE * ONE",
    "SEVEN = SIX * ONE",
    "EIGHT = SEVEN * ONE",
    "NINE = EIGHT * ONE",
    "TEN = NINE * ONE",
    "ELEVEN = TEN * ONE",
    "TWELVE = ELEVEN * ONE",
    "THIRTEEN = TWELVE * ONE",
    "FOURTEEN = THIRTEEN * ONE",
    "FIFTEEN = FOURTEEN * ONE",
    "SIXTEEN = FIFTEEN * ONE",
    "SEVENTEEN = SIXTEEN * ONE",
    "EIGHTEEN = SEVENTEEN * ONE",
    "NINETEEN = EIGHTEEN * ONE",
    "TWENTY = NINETEEN * ONE",
    "TWENTY_ONE = TWENTY * ONE",
    "TWENTY_TWO = TWENTY_ONE * ONE",
    "TWENTY_THREE = TWENTY_TWO * ONE",
    "TWENTY_FOUR = TWENTY_THREE * ONE",
    "TWENTY_FIVE = TWENTY_FOUR * ONE",
    "TWENTY_SIX = TWENTY_FIVE * ONE",
    "TWENTY_SEVEN = TWENTY_SIX * ONE",
    "TWENTY_EIGHT = TWENTY_SEVEN * ONE",
    "TWENTY_NINE = TWENTY_EIGHT * ONE",
    "THIRTY = TWENTY_NINE * ONE",
    "THIRTY_ONE = THIRTY * ONE",
    "THIRTY_TWO = THIRTY_ONE * ONE",
    "THIRTY_THREE = THIRTY_TWO * ONE",
    "THIRTY_FOUR = THIRTY_THREE * ONE",
    "THIRTY_FIVE = THIRTY_FOUR * ONE",
    "THIRTY_SIX = THIRTY_FIVE * ONE",
    "THIRTY_SEVEN = THIRTY_SIX * ONE",
    "THIRTY_EIGHT = THIRTY_SEVEN * ONE",
    "THIRTY_NINE = THIRTY_EIGHT * ONE",
    "FORTY = THIRTY_NINE * ONE",
    "FORTY_ONE = FORTY * ONE",
    "FORTY_TWO = FORTY_ONE * ONE",
    "FORTY_THREE = FORTY_TWO * ONE",
    "FORTY_FOUR = FORTY_THREE * ONE",
    "FORTY_FIVE = FORTY_FOUR * ONE",
    "FORTY_SIX = FORTY_FIVE * ONE",
    "FORTY_SEVEN = FORTY_SIX * ONE",
    "FORTY_EIGHT = FORTY_SEVEN * ONE",
    "FORTY_NINE = FORTY_EIGHT * ONE",
    "FIFTY = FORTY_NINE * ONE",
    "FIFTY_ONE = FIFTY * ONE",
    "FIFTY_TWO = FIFTY_ONE * ONE",
    "FIFTY_THREE = FIFTY_TWO * ONE",
    "FIFTY_FOUR = FIFTY_THREE * ONE",
    "FIFTY_FIVE = FIFTY_FOUR * ONE",
    "FIFTY_SIX = FIFTY_FIVE * ONE",
    "FIFTY_SEVEN = FIFTY_SIX * ONE",
    "FIFTY_EIGHT = FIFTY_SEVEN * ONE",
    "FIFTY_NINE = FIFTY_EIGHT * ONE",
    "SIXTY = FIFTY_NINE * ONE",
    "SIXTY_ONE = SIXTY * ONE",
    "SIXTY_TWO = SIXTY_ONE * ONE",
    "SIXTY_THREE = SIXTY_TWO * ONE",
    "SIXTY_FOUR = SIXTY_THREE * ONE",
    "SIXTY_FIVE = SIXTY_FOUR * ONE",
    "SIXTY_SIX = SIXTY_FIVE * ONE",
    "SIXTY_SEVEN = SIXTY_SIX * ONE",
    "SIXTY_EIGHT = SIXTY_SEVEN * ONE",
    "SIXTY_NINE = SIXTY_EIGHT * ONE",
    "SEVENTY = SIXTY_NINE * ONE",
    "SEVENTY_ONE = SEVENTY * ONE",
    "SEVENTY_TWO = SEVENTY_ONE * ONE",
    "SEVENTY_THREE = SEVENTY_TWO * ONE",
    "SEVENTY_FOUR = SEVENTY_THREE * ONE",
    "SEVENTY_FIVE = SEVENTY_FOUR * ONE",
    "SEVENTY_SIX = SEVENTY_FIVE * ONE",
    "SEVENTY_SEVEN = SEVENTY_SIX * ONE",
    "SEVENTY_EIGHT = SEVENTY_SEVEN * ONE",
    "SEVENTY_NINE = SEVENTY_EIGHT * ONE",
    "EIGHTY = SEVENTY_NINE * ONE",
    "EIGHTY_ONE = EIGHTY * ONE",
    "EIGHTY_TWO = EIGHTY_ONE * ONE",
    "EIGHTY_THREE = EIGHTY_TWO * ONE",
    "EIGHTY_FOUR = EIGHTY_THREE * ONE",
    "EIGHTY_FIVE = EIGHTY_FOUR * ONE",
    "EIGHTY_SIX = EIGHTY_FIVE * ONE",
    "EIGHTY_SEVEN = EIGHTY_SIX * ONE",
    "EIGHTY_EIGHT = EIGHTY_SEVEN * ONE",
    "EIGHTY_NINE = EIGHTY_EIGHT * ONE",
    "NINETY = EIGHTY_NINE * ONE",
    "NINETY_ONE = NINETY * ONE",
    "NINETY_TWO = NINETY_ONE * ONE",
    "NINETY_THREE = NINETY_TWO * ONE",
    "NINETY_FOUR = NINETY_THREE * ONE",
    "NINETY_FIVE = NINETY_FOUR * ONE",
    "NINETY_SIX = NINETY_FIVE * ONE",
    "NINETY_SEVEN = NINETY_SIX * ONE",
    "NINETY_EIGHT = NINETY_SEVEN * ONE",
    "NINETY_NINE = NINETY_EIGHT * ONE",
    "HUNDRED = NINETY_NINE * ONE",
]

# Set up SP space
SP_SPACE = spa.Vocabulary(
    N_DIMENSIONS,
    pointer_gen=np.random.RandomState(SEED),
)
SP_SPACE.populate(";".join(object_tags + feature_tags + colour_tags + number_tags))

# Get the SPs for each type of represented attribute
OBJ_SPS = SP_SPACE.vectors[:(MAX_N_OBJECTS)]
FEATURE_SPS = SP_SPACE.vectors[(MAX_N_OBJECTS):(MAX_N_OBJECTS + N_FEATURES)]
COLOUR_SPS = SP_SPACE.vectors[(MAX_N_OBJECTS + N_FEATURES):(MAX_N_OBJECTS + N_FEATURES + N_COLOURS)]
NUMBER_SPS = SP_SPACE.vectors[(MAX_N_OBJECTS + N_FEATURES + N_COLOURS):]

# Compatibility wrapper: provides old RandomSSPSpace interface over new sspspace API
# -----------------------------------------------------------------------------
class CompatSSPSpace:
    """
    Wraps the new sspspace API (SSPEncoder + top-level bind/invert + SSPSimilarityDecoder)
    to match the interface expected by this codebase (bind/invert/decode/get_sample_pts_and_ssps
    as methods on the space object).
    """
    def __init__(self, domain_dim, ssp_dim, domain_bounds, length_scale, rng, **kwargs):
        # kwargs absorbs deprecated params like 'sampler' without error
        self._encoder = sspspace.RandomSSPSpace(
            domain_dim=domain_dim,
            ssp_dim=ssp_dim,
            length_scale=length_scale,
            rng=rng,
        )
        self._domain_bounds = np.array(domain_bounds)
        self._decoder = None

    def encode(self, x):
        """Encode point(s). Input (2,) or (N,2); returns (1,D) or (N,D)."""
        return self._encoder.encode(np.atleast_2d(x))

    def bind(self, a, b):
        """Circular convolution (HRR binding). Pure numpy, no SSP object needed."""
        a_flat = np.asarray(a, dtype=float).flatten()
        b_flat = np.asarray(b, dtype=float).flatten()
        return np.real(np.fft.ifft(np.fft.fft(a_flat) * np.fft.fft(b_flat)))

    def invert(self, a):
        """HRR approximate inverse: [a[0], a[n-1], a[n-2], ..., a[1]]. Pure numpy."""
        a_flat = np.asarray(a, dtype=float).flatten()
        return a_flat[-np.arange(len(a_flat))]

    def get_sample_pts_and_ssps(self, num_points_per_dim):
        """Build uniform sample grid over domain; initialises the decoder."""
        xs = np.linspace(self._domain_bounds[0, 0], self._domain_bounds[0, 1], num_points_per_dim)
        ys = np.linspace(self._domain_bounds[1, 0], self._domain_bounds[1, 1], num_points_per_dim)
        XX, YY = np.meshgrid(xs, ys)
        sim_xs = np.column_stack([XX.ravel(), YY.ravel()])  # (N*N, 2)
        sim_ssps = self._encoder.encode(sim_xs)              # (N*N, D)
        self._decoder = sspspace.SSPSimilarityDecoder(sim_xs, sim_ssps, self._encoder)
        return (sim_xs, sim_ssps)  # kept for API compatibility; decode ignores it

    def decode(self, x, samples=None):
        """Decode SSP to nearest sample point. Returns shape (1, 2)."""
        if self._decoder is None:
            raise RuntimeError("Call get_sample_pts_and_ssps before decode.")
        result = self._decoder.decode(np.atleast_2d(x))
        return np.atleast_2d(result)  # ensure (1, 2) as callers expect
# -----------------------------------------------------------------------------

# Set up SSP space
DOMAIN_BOUNDS = np.array([[-MAX_GRID_SIZE / 2, MAX_GRID_SIZE / 2], [-MAX_GRID_SIZE / 2, MAX_GRID_SIZE / 2]])
SSP_SPACE = CompatSSPSpace(
    domain_dim=2,
    ssp_dim=N_DIMENSIONS,
    domain_bounds=DOMAIN_BOUNDS,
    length_scale=LENGTH_SCALE,
    sampler="norm",
    rng=np.random.default_rng(SEED),
)

# Cache samples to use for centre decoding
SAMPLES = SSP_SPACE.get_sample_pts_and_ssps(num_points_per_dim=(2 * MAX_GRID_SIZE + 1))
# -----------------------------------------------------------------------------


# Utilities to translate VSA representations to meaning
# -----------------------------------------------------------------------------
# Mapping from ARC colours to printable symbols
COLOUR_MAP = {
    0: "⬛",  # black
    1: "🟦",  # blue
    2: "🟥",  # red
    3: "🟩",  # green
    4: "🟨",  # yellow
    5: "⬜",  # grey  (white square — closest reliably 2-wide option)
    6: "🟣",  # pink  (purple circle — not perfect but consistently wide)
    7: "🟧",  # orange
    8: "🔷",  # cyan  (large blue diamond — reliably 2-wide)
    9: "🟫",  # maroon (brown square)
}

# Mapping from ARC colours to displayable colours
DISPLAY_COLOURS  = {
    "Black": "#000000",
    "Blue": "#0074D9",
    "Red": "#FF4136",
    "Green": "#2ECC40",
    "Yellow": "#FFDC00",
    "Grey": "#AAAAAA",
    "Pink": "#F012BE",
    "Orange": "#FF851B",
    "Cyan": "#7FDBFF",
    "Purple": "#870C25",
}
for key in DISPLAY_COLOURS.keys():
    DISPLAY_COLOURS[key] = [val / 255.0 for val in ImageColor.getcolor(DISPLAY_COLOURS[key], "RGB")]

# Mapping from integers to semantic pointer names and vice versa
NUMBER_MAP = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
    "TWENTY": 20,
    "TWENTY_ONE": 21,
    "TWENTY_TWO": 22,
    "TWENTY_THREE": 23,
    "TWENTY_FOUR": 24,
    "TWENTY_FIVE": 25,
    "TWENTY_SIX": 26,
    "TWENTY_SEVEN": 27,
    "TWENTY_EIGHT": 28,
    "TWENTY_NINE": 29,
    "THIRTY": 30,
    "THIRTY_ONE": 31,
    "THIRTY_TWO": 32,
    "THIRTY_THREE": 33,
    "THIRTY_FOUR": 34,
    "THIRTY_FIVE": 35,
    "THIRTY_SIX": 36,
    "THIRTY_SEVEN": 37,
    "THIRTY_EIGHT": 38,
    "THIRTY_NINE": 39,
    "FORTY": 40,
    "FORTY_ONE": 41,
    "FORTY_TWO": 42,
    "FORTY_THREE": 43,
    "FORTY_FOUR": 44,
    "FORTY_FIVE": 45,
    "FORTY_SIX": 46,
    "FORTY_SEVEN": 47,
    "FORTY_EIGHT": 48,
    "FORTY_NINE": 49,
    "FIFTY": 50,
    "FIFTY_ONE": 51,
    "FIFTY_TWO": 52,
    "FIFTY_THREE": 53,
    "FIFTY_FOUR": 54,
    "FIFTY_FIVE": 55,
    "FIFTY_SIX": 56,
    "FIFTY_SEVEN": 57,
    "FIFTY_EIGHT": 58,
    "FIFTY_NINE": 59,
    "SIXTY": 60,
    "SIXTY_ONE": 61,
    "SIXTY_TWO": 62,
    "SIXTY_THREE": 63,
    "SIXTY_FOUR": 64,
    "SIXTY_FIVE": 65,
    "SIXTY_SIX": 66,
    "SIXTY_SEVEN": 67,
    "SIXTY_EIGHT": 68,
    "SIXTY_NINE": 69,
    "SEVENTY": 70,
    "SEVENTY_ONE": 71,
    "SEVENTY_TWO": 72,
    "SEVENTY_THREE": 73,
    "SEVENTY_FOUR": 74,
    "SEVENTY_FIVE": 75,
    "SEVENTY_SIX": 76,
    "SEVENTY_SEVEN": 77,
    "SEVENTY_EIGHT": 78,
    "SEVENTY_NINE": 79,
    "EIGHTY": 80,
    "EIGHTY_ONE": 81,
    "EIGHTY_TWO": 82,
    "EIGHTY_THREE": 83,
    "EIGHTY_FOUR": 84,
    "EIGHTY_FIVE": 85,
    "EIGHTY_SIX": 86,
    "EIGHTY_SEVEN": 87,
    "EIGHTY_EIGHT": 88,
    "EIGHTY_NINE": 89,
    "NINETY": 90,
    "NINETY_ONE": 91,
    "NINETY_TWO": 92,
    "NINETY_THREE": 93,
    "NINETY_FOUR": 94,
    "NINETY_FIVE": 95,
    "NINETY_SIX": 96,
    "NINETY_SEVEN": 97,
    "NINETY_EIGHT": 98,
    "NINETY_NINE": 99,
    "HUNDRED": 100,
    0: "ZERO",
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "FIVE",
    6: "SIX",
    7: "SEVEN",
    8: "EIGHT",
    9: "NINE",
    10: "TEN",
    11: "ELEVEN",
    12: "TWELVE",
    13: "THIRTEEN",
    14: "FOURTEEN",
    15: "FIFTEEN",
    16: "SIXTEEN",
    17: "SEVENTEEN",
    18: "EIGHTEEN",
    19: "NINETEEN",
    20: "TWENTY",
    21: "TWENTY_ONE",
    22: "TWENTY_TWO",
    23: "TWENTY_THREE",
    24: "TWENTY_FOUR",
    25: "TWENTY_FIVE",
    26: "TWENTY_SIX",
    27: "TWENTY_SEVEN",
    28: "TWENTY_EIGHT",
    29: "TWENTY_NINE",
    30: "THIRTY",
    31: "THIRTY_ONE",
    32: "THIRTY_TWO",
    33: "THIRTY_THREE",
    34: "THIRTY_FOUR",
    35: "THIRTY_FIVE",
    36: "THIRTY_SIX",
    37: "THIRTY_SEVEN",
    38: "THIRTY_EIGHT",
    39: "THIRTY_NINE",
    40: "FORTY",
    41: "FORTY_ONE",
    42: "FORTY_TWO",
    43: "FORTY_THREE",
    44: "FORTY_FOUR",
    45: "FORTY_FIVE",
    46: "FORTY_SIX",
    47: "FORTY_SEVEN",
    48: "FORTY_EIGHT",
    49: "FORTY_NINE",
    50: "FIFTY",
    51: "FIFTY_ONE",
    52: "FIFTY_TWO",
    53: "FIFTY_THREE",
    54: "FIFTY_FOUR",
    55: "FIFTY_FIVE",
    56: "FIFTY_SIX",
    57: "FIFTY_SEVEN",
    58: "FIFTY_EIGHT",
    59: "FIFTY_NINE",
    60: "SIXTY",
    61: "SIXTY_ONE",
    62: "SIXTY_TWO",
    63: "SIXTY_THREE",
    64: "SIXTY_FOUR",
    65: "SIXTY_FIVE",
    66: "SIXTY_SIX",
    67: "SIXTY_SEVEN",
    68: "SIXTY_EIGHT",
    69: "SIXTY_NINE",
    70: "SEVENTY",
    71: "SEVENTY_ONE",
    72: "SEVENTY_TWO",
    73: "SEVENTY_THREE",
    74: "SEVENTY_FOUR",
    75: "SEVENTY_FIVE",
    76: "SEVENTY_SIX",
    77: "SEVENTY_SEVEN",
    78: "SEVENTY_EIGHT",
    79: "SEVENTY_NINE",
    80: "EIGHTY",
    81: "EIGHTY_ONE",
    82: "EIGHTY_TWO",
    83: "EIGHTY_THREE",
    84: "EIGHTY_FOUR",
    85: "EIGHTY_FIVE",
    86: "EIGHTY_SIX",
    87: "EIGHTY_SEVEN",
    88: "EIGHTY_EIGHT",
    89: "EIGHTY_NINE",
    90: "NINETY",
    91: "NINETY_ONE",
    92: "NINETY_TWO",
    93: "NINETY_THREE",
    94: "NINETY_FOUR",
    95: "NINETY_FIVE",
    96: "NINETY_SIX",
    97: "NINETY_SEVEN",
    98: "NINETY_EIGHT",
    99: "NINETY_NINE",
    100: "HUNDRED",
}
# -----------------------------------------------------------------------------


# Grid size operations
# -----------------------------------------------------------------------------
# Query for bundle height
def query_height(bundle_ssp):
    return normalize(
        SSP_SPACE.bind(
            bundle_ssp,
            SSP_SPACE.invert(SP_SPACE["HEIGHT"].v),
        ).flatten()
    )


# Query for bundle width
def query_width(bundle_ssp):
    return normalize(
        SSP_SPACE.bind(
            bundle_ssp,
            SSP_SPACE.invert(SP_SPACE["WIDTH"].v),
        ).flatten()
    )


# Bundle the grid dimensions together in a slot-filler representation
def bundle_size(height, width):
    return normalize(
        SSP_SPACE.bind(
            SP_SPACE["HEIGHT"].v,
            SP_SPACE[NUMBER_MAP[height]].v,
        ) + SSP_SPACE.bind(
            SP_SPACE["WIDTH"].v,
            SP_SPACE[NUMBER_MAP[width]].v,
        )
    )


# Clean up SP to a number SP
def cleanup_number(number_ssp):
    cleaned_number_index = np.argmax(number_ssp.flatten() @ NUMBER_SPS.T)
    cleaned_number_ssp = NUMBER_SPS[cleaned_number_index].flatten()
    return cleaned_number_index, cleaned_number_ssp


# Clean up SP to two number SPs representing the grid dimensions
def cleanup_size(size_ssp):
    cleaned_height_index, cleaned_height_ssp = cleanup_number(query_height(size_ssp))
    cleaned_width_index, cleaned_width_ssp = cleanup_number(query_width(size_ssp))
    return cleaned_height_index, cleaned_height_ssp, cleaned_width_index, cleaned_width_ssp
# -----------------------------------------------------------------------------


# Clean up operations
# -----------------------------------------------------------------------------
# Clean up SP to a colour SP
def cleanup_colour(colour_ssp):
    cleaned_colour_index = np.argmax(colour_ssp.flatten() @ COLOUR_SPS.T)
    cleaned_colour_ssp = COLOUR_SPS[cleaned_colour_index].flatten()
    return cleaned_colour_index, cleaned_colour_ssp


# Clean up SP to an SSP point
def cleanup_centre(centre_ssp):
    cleaned_centre_index = SSP_SPACE.decode(centre_ssp.reshape((1, N_DIMENSIONS)), samples=SAMPLES)
    cleaned_centre_ssp = SSP_SPACE.encode(cleaned_centre_index).flatten()
    return cleaned_centre_index, cleaned_centre_ssp


# Clean up SP to an SSP bundle centred at the origin
def cleanup_shape(shape_ssp, centre_ssp, size_ssp):
    # Convert known information into a position representation
    cleaned_centre_index, cleaned_centre_ssp = cleanup_centre(centre_ssp)
    position_ssp = SSP_SPACE.bind(shape_ssp, cleaned_centre_ssp)
    cleaned_position_indices, cleaned_position_ssp = cleanup_position(position_ssp, size_ssp)

    # Centre this representation back at the origin
    cleaned_shape_indices = []
    for point in cleaned_position_indices:
        cleaned_shape_indices.append((point[0] - cleaned_centre_index[0][1], point[1] + cleaned_centre_index[0][0]))
    cleaned_shape_ssp = normalize(SSP_SPACE.bind(cleaned_position_ssp, SSP_SPACE.invert(cleaned_centre_ssp)).flatten())
    return cleaned_shape_indices, cleaned_shape_ssp

# permissive candidate cutoff: below the lowest true-pixel similarity (observed >=0.08
# even for 49-px dense blocks) and above pure cross-talk noise. Over-inclusion only
# costs a little NNLS time; it must never exclude a real pixel.
POSITION_CANDIDATE_THRESHOLD = 0.02
# keep candidate cells whose NNLS weight exceeds this fraction of the max weight.
POSITION_WEIGHT_FRACTION = 0.40
 
# cache of per-grid-size cell-SSP dictionaries (built once per (n_rows, n_cols))
_POSITION_CELL_CACHE = {}
 
def _position_cell_dictionary(n_rows, n_cols):
    key = (n_rows, n_cols)
    if key not in _POSITION_CELL_CACHE:
        cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]
        dictionary = np.array([
            SSP_SPACE.encode([j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]).flatten()
            for (i, j) in cells
        ])  # (n_cells, N_DIMENSIONS)
        _POSITION_CELL_CACHE[key] = (cells, dictionary)
    return _POSITION_CELL_CACHE[key]

# Clean up SP to an SSP bundle
def cleanup_position(position_ssp, size_ssp):
    n_rows, _, n_cols, _ = cleanup_size(size_ssp)
    cells, dictionary = _position_cell_dictionary(n_rows, n_cols)
    query = np.asarray(position_ssp).flatten()
 
    cleaned_position_indices = []
    cleaned_position_ssp = np.zeros((N_DIMENSIONS))
 
    # Stage 1 — cheap similarity field -> permissive candidate superset
    sims = dictionary @ query
    candidate = np.where(sims > POSITION_CANDIDATE_THRESHOLD)[0]
    if candidate.size == 0:
        return cleaned_position_indices, cleaned_position_ssp
 
    # Stage 2 — recover the support over candidates by non-negative least squares
    weights, _ = nnls(dictionary[candidate].T, query)
    w_max = weights.max() if weights.size else 0.0
    if w_max <= 0.0:
        return cleaned_position_indices, cleaned_position_ssp
 
    for t, k in enumerate(candidate):
        if weights[t] > POSITION_WEIGHT_FRACTION * w_max:
            i, j = cells[k]
            cleaned_position_indices.append((i, j))
            cleaned_position_ssp += SSP_SPACE.encode(
                [j - (n_cols - 1) / 2, (n_rows - 1) / 2 - i]
            ).flatten()
 
    return cleaned_position_indices, cleaned_position_ssp



# Reconstruct pixel grid from scene SSP bundle
def decode_grid(obj_ssps, size_ssp):
    n_rows, _, n_cols, _ = cleanup_size(size_ssp)
    grid = np.zeros((n_rows, n_cols))

    for obj in range(len(obj_ssps)):
        colour = cleanup_colour(SSP_SPACE.bind(obj_ssps[obj], SSP_SPACE.invert(SP_SPACE["COLOUR"].v)))[0]
        centre = cleanup_centre(SSP_SPACE.bind(obj_ssps[obj], SSP_SPACE.invert(SP_SPACE["CENTRE"].v)))[1]
        shape = SSP_SPACE.bind(obj_ssps[obj], SSP_SPACE.invert(SP_SPACE["SHAPE"].v))
        position = cleanup_position(SSP_SPACE.bind(centre, shape), size_ssp)[0]

        for pixel in position:
            grid[pixel[0]][pixel[1]] = colour
    return grid
# -----------------------------------------------------------------------------


# General utilities for VSA representations
# -----------------------------------------------------------------------------
# Make a SP/SSP unit length
def normalize(vector):
    # Check for divide-by-zero case
    if (vector == 0).all():
        print("WARNING: Normalizing the zero vector.")
        return vector.flatten()
    return vector.flatten() / np.linalg.norm(vector)


# Create a specialized kernel for spatial reasoning
def centre_kernel(var):
    return np.exp(-((var[0] ** 2) + (var[1] ** 2) / 100)) + np.exp(-((var[0] ** 2) / 100 + (var[1] ** 2))) + np.exp(-(var[0] ** 2 + var[1] ** 2) / 10) * 5
N_SAMPLES = 100
SAMPLE_RANGE = MAX_GRID_SIZE / 2
samples = np.linspace(-np.sqrt(SAMPLE_RANGE), np.sqrt(SAMPLE_RANGE), N_SAMPLES + 1)
samples = (2 * (samples > 0) - 1) * (samples ** 2)
xs, ys = np.meshgrid(samples, samples)
points = np.vstack([xs.ravel(), ys.ravel()]).T
Y = np.array([centre_kernel(point) for point in points]) / 100
A = SSP_SPACE.encode(points)
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1, fit_intercept=True)
ridge_model.fit(A, Y)
soln = normalize(ridge_model.coef_)

# # Displays visualization of induced kernel
# import matplotlib.pyplot as plt
# SSP_SPACE.similarity_plot(soln, 100)
# plt.show()

# Scales centre representations to be more similar to each other
def scale_centre(centre_ssp):
    return SSP_SPACE.bind(centre_ssp, soln).flatten()


# Dynamically compute optimal decoding threshold
def compute_threshold(sims):
    # # Prints pixel-by-pixel similarity
    # print("Similarity grid:")
    # for i in range(len(sims)):
    #     for j in range(len(sims[0])):
    #         print(f"{sims[i][j]:.3f}", end=" ")
    #         # print(f"{sims[i][j] ** 2:.3f}", end=" ")
    #         # print(f"{abs(sims[i][j]):.3f}", end=" ")
    #     print()
    # print()

    # Compute threshold as local minimum of pdf of mixture of two Gaussians
    sims = sorted([0] + [sim for temp in sims for sim in temp])
    gmm = GaussianMixture(n_components=2, random_state=SEED)
    gmm.fit(np.array(sims).reshape(-1, 1))
    domain = np.linspace(sims[0], sims[-1], 100).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(domain))
    threshold = 0.1
    for i in range(1, len(pdf) - 1):
        if (pdf[i-1] > pdf[i]) and (pdf[i+1] > pdf[i]):
            threshold = domain[i][0]
    threshold = np.clip(threshold, 0.01, 0.5)

    # # Prints computed dynamic threshold
    # print("Computed Threshold:")
    # print(threshold)
    # print()
    return threshold
# -----------------------------------------------------------------------------


def main():
    return


if __name__ == "__main__":
    main()
