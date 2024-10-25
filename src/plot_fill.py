import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"]=300
plt.rcParams.update({'font.size': 20})
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
#dir path
from TQPDEinput import get_dir_QPDE_path, get_hub_group_project
hub, group, project = get_hub_group_project()
dir_QPDE_path = get_dir_QPDE_path()

#input ============================================
service = QiskitRuntimeService(channel='ibm_quantum',instance=f'{hub}/{group}/{project}')

gap_list = []
shots_list = []
initial_mu_list = []
initial_var_list = []
png_save_name_list = []
prob_list_list = []
mu_list_list = []
var_list_list = []
label_list = []

#Hubbard 4sites = 9 qubits
#aer
gap = 0.25360840680916397
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
png_save_name="TQPDE_Hubbard_4site_aer"
label = "Exact"
prob_list=[[0.2399, 0.3297, 0.4234, 0.5264, 0.614, 0.7096, 0.7929, 0.8717, 0.9219, 0.9638, 0.9851, 0.9877, 0.9701, 0.9308, 0.8757, 0.8068, 0.7265, 0.6316, 0.532, 0.4326, 0.3477], [0.2353, 0.322, 0.4111, 0.5144, 0.6093, 0.7148, 0.7908, 0.8612, 0.9206, 0.9601, 0.9826, 0.9816, 0.9621, 0.9233, 0.8616, 0.7961, 0.7179, 0.6206, 0.5183, 0.4247, 0.3281], [0.3266, 0.4106, 0.4996, 0.5915, 0.6807, 0.7514, 0.8321, 0.8857, 0.935, 0.9655, 0.9837, 0.9845, 0.9687, 0.9369, 0.887, 0.823, 0.7574, 0.6769, 0.5905, 0.503, 0.4097], [0.3307, 0.4142, 0.5103, 0.5853, 0.675, 0.7587, 0.8173, 0.8828, 0.9264, 0.9595, 0.9772, 0.9783, 0.9627, 0.9285, 0.8891, 0.8236, 0.7639, 0.6754, 0.6018, 0.5174, 0.4229], [0.3371, 0.4132, 0.5057, 0.5861, 0.6831, 0.761, 0.8275, 0.8886, 0.9302, 0.963, 0.9788, 0.9788, 0.9619, 0.9292, 0.8842, 0.8224, 0.7524, 0.6756, 0.5999, 0.5054, 0.4196], [0.3061, 0.3667, 0.4331, 0.5022, 0.562, 0.6159, 0.6557, 0.703, 0.7421, 0.7518, 0.7523, 0.748, 0.7327, 0.6992, 0.6684, 0.609, 0.5625, 0.4913, 0.429, 0.3625, 0.3065]]
mu_list=[0.087254, 0.155217, 0.195128, 0.21851, 0.224224, 0.224205]
var_list=[2.529954, 1.30437, 0.531314, 0.119091, 0.007183, 2.9e-05]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#torino
gap = 0.25360840680916397
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
label = "Exact"
png_save_name="TQPDE_Hubbard_4site_torino"
prob_list=[[0.2004316, 0.2969795, 0.3466836, 0.419081, 0.3817277, 0.5402001, 0.5925794, 0.5859592, 0.5874667, 0.6598073, 0.7011763, 0.7199041, 0.7536282, 0.6776555, 0.6833155, 0.6784714, 0.4825175, 0.4587122, 0.4486624, 0.3593797, 0.2855662], [0.1875257, 0.1435095, 0.291038, 0.3239539, 0.320991, 0.2824695, 0.3906277, 0.3806548, 0.3772573, 0.3648344, 0.4600859, 0.4159838, 0.3957955, 0.4391696, 0.3929449, 0.3773512, 0.3064179, 0.2915458, 0.2304445, 0.3084967, 0.1589335], [0.0369415, 0.1088667, 0.0493224, 0.0817655, 0.1262451, 0.1037546, 0.1068703, 0.1550933, 0.1175275, 0.1277003, 0.1613339, 0.1416166, 0.1221776, 0.1164565, 0.1387359, 0.0775085, 0.1082161, 0.0727606, 0.0829578, 0.0439647, 0.0772279], [0.0085563, 0.0163798, 0.0114759, 0.0097067, 0.0116272, 0.021372, 0.0180082, 0.0156924, 0.017961, 0.0199603, 0.0193646, 0.0214207, 0.0173261, 0.0218242, 0.0206043, 0.0205596, 0.020421, 0.0190748, 0.0131835, 0.0099612, 0.0178128], [0.0031391, 0.0021652, 0.0038414, 0.0030079, 0.0021492, 0.0021859, 0.0020134, 0.0038393, 0.0034811, 0.0045834, 0.0031197, 0.0024619, 0.0016683, 0.0029915, 0.0031103, 0.0048005, 0.0027938, 0.0024807, 0.0033065, 0.0026695, 0.0032737], [0.0023011, 0.0015255, 0.0031215, 0.0023715, 0.0017037, 0.0016639, 0.0032271, 0.0029255, 0.0021066, 0.0025607, 0.0029846, 0.0028262, 0.0034777, 0.0024911, 0.0008724, 0.0028202, 0.0031835, 0.0025642, 0.0031342, 0.0022932, 0.0024508]]
mu_list=[0.117961, 0.156813, 0.135466, 0.211911, 0.249532, 0.294099]
var_list=[2.662446, 1.675646, 0.794743, 0.33691, 0.261818, 0.128824]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#Hubbard 10 sites 21 qubits
initial_mu = 0.0  
initial_var = 4.0  
gap = 0.12579074827220627 #MPS
shots = 1e4
label = "MPS"
png_save_name="TQPDE_Hubbard_10site_torino"
prob_list=[[0.1103878, 0.3547873, 0.3484911, 0.3894695, 0.4212925, 0.401492, 0.4949778, 0.5523202, 0.5018016, 0.5527283, 0.4666846, 0.5387414, 0.461883, 0.5074201, 0.5379993, 0.4955716, 0.4838192, 0.4695884, 0.4160272, 0.5673845, 0.4148843], [0.1484441, 0.4111609, 0.3918217, 0.4307267, 0.484509, 0.4515986, 0.4240917, 0.4720992, 0.6367651, 0.395333, 0.5181763, 0.5279819, 0.46657, 0.4879844, 0.5202316, 0.4895594, 0.4241811, 0.4190032, 0.1441749, 0.1155689, 0.1274278], [0.0952327, 0.0621194, 0.0883297, 0.0715439, 0.106589, 0.131037, 0.1262933, 0.1186544, 0.1379053, 0.1253414, 0.0963333, 0.122093, 0.0988258, 0.0900222, 0.0939458, 0.0754474, 0.0743136, 0.0808304, 0.0761777, 0.0741975, 0.0092308], [0.2592773, 0.3484708, 0.3007368, 0.3024403, 0.3463455, 0.4334486, 0.4874403, 0.4793117, 0.6663706, 0.6514039, 0.7576993, 0.7121702, 0.6305373, 0.6357716, 0.617277, 0.6808767, 0.6785311, 0.4942543, 0.52553, 0.4796674, 0.3422712], [0.1375894, 0.3782801, 0.3482362, 0.3471331, 0.494217, 0.4775928, 0.465435, 0.6004612, 0.617613, 0.5485336, 0.5637687, 0.6461894, 0.651813, 0.4321173, 0.5003088, 0.4738097, 0.4322013, 0.442397, 0.3615894, 0.1296725, 0.1346961]]
mu_list=[0.169285, 0.060016, -0.099298, 0.039685, 0.025935]
var_list=[3.286265, 2.136104, 1.252131, 0.535632, 0.10892]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#Hubbard 16 sites 33 qubits
initial_mu = 0.0  
initial_var = 4.0  
gap = 0.08417826689489516 #MPS
shots = 1e4
label = "MPS"
png_save_name="TQPDE_Hubbard_16site_torino"
prob_list=[[0.0004254, 0.0, 0.000485, 0.0, 0.0, 0.0003894, 0.0004467, 0.0003833, 0.0003693, 0.0, 6.77e-05, 0.000306, 0.0, 0.000495, 0.0003833, 1e-07, 0.0004752, 0.0, 0.0004286, 7.78e-05, 0.0], [0.0020986, 0.0021315, 0.0019934, 0.0030813, 0.0019801, 0.0041263, 0.0052681, 0.0035349, 0.0045556, 0.0028059, 0.0066411, 0.001772, 0.0033728, 0.003335, 0.0048223, 0.0041647, 0.001361, 0.0020315, 0.0031151, 0.0019761, 0.0022711], [0.0015672, 0.0020267, 0.0036932, 0.0061488, 0.001836, 0.0018087, 0.003636, 0.0052128, 0.005059, 0.0094145, 0.007248, 0.008155, 0.0067089, 0.0070195, 0.0098155, 0.0050546, 0.0115261, 0.0042879, 0.0046809, 0.003656, 0.0052634], [0.0017272, 0.001633, 0.0035767, 0.0030237, 0.0044143, 0.0061168, 0.0042854, 0.0077844, 0.003287, 0.0034796, 0.0041123, 0.0085503, 0.0045944, 0.0016501, 0.0052082, 0.0033971, 0.0051351, 0.0051247, 0.0039649, 0.001804, 0.0018226], [0.0017285, 0.0048584, 0.0073514, 0.0071927, 0.0042412, 0.004196, 0.0072444, 0.0030321, 0.0088073, 0.0118675, 0.0153729, 0.0105258, 0.0061782, 0.0057481, 0.0067604, 0.0033409, 0.0034447, 0.0043471, 0.0028803, 0.0018811, 1.31e-05]]
mu_list=[-0.168603, -0.211206, 0.085966, 0.082711, 0.043234]
var_list=[3.489482, 2.390518, 1.23543, 0.538085, 0.072725]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#Hubbard 4sites = 9 qubits vac
#aer
#QPDE
gap = -20.91149746860633
shots = 1e4
# initial value
initial_mu = 20.23102405089335 # value of MPS maxdim 2
initial_var = 4.0
png_save_name="TQPDE_vac_Hubbard_4site_aer"
label = "Exact"
prob_list=[[0.1467, 0.2378, 0.3167, 0.4092, 0.5089, 0.6088, 0.7044, 0.7866, 0.8555, 0.9201, 0.9543, 0.9797, 0.9824, 0.967, 0.9343, 0.8815, 0.8102, 0.7361, 0.6356, 0.5442, 0.4397], [0.1613, 0.2407, 0.3254, 0.4202, 0.5095, 0.6186, 0.7148, 0.7868, 0.8526, 0.9168, 0.9503, 0.9698, 0.9722, 0.9429, 0.908, 0.8532, 0.7862, 0.7045, 0.6123, 0.5093, 0.4055], [0.2461, 0.3298, 0.4143, 0.514, 0.5928, 0.6697, 0.753, 0.8253, 0.8888, 0.932, 0.9591, 0.9741, 0.9712, 0.9592, 0.924, 0.8755, 0.8213, 0.7486, 0.6688, 0.5868, 0.4966], [0.2387, 0.3148, 0.405, 0.4947, 0.5947, 0.6808, 0.7515, 0.8198, 0.8905, 0.9303, 0.9643, 0.9802, 0.9796, 0.9629, 0.9275, 0.8862, 0.8176, 0.7563, 0.6718, 0.5768, 0.4979], [0.2586, 0.3373, 0.4283, 0.5207, 0.6025, 0.6875, 0.7656, 0.8208, 0.8874, 0.9202, 0.952, 0.9689, 0.9646, 0.9491, 0.9098, 0.8659, 0.8055, 0.7383, 0.6641, 0.5777, 0.4869], [0.2169, 0.2811, 0.3686, 0.4584, 0.5283, 0.6286, 0.7045, 0.7804, 0.8392, 0.8939, 0.9312, 0.9574, 0.9649, 0.9607, 0.9389, 0.9023, 0.856, 0.7921, 0.7177, 0.6397, 0.564]]
mu_list=[20.48597, 20.671141, 20.786634, 20.849894, 20.864647, 20.866034]
var_list=[2.518592, 1.294801, 0.526082, 0.11232, 0.006414, 2.2e-05]

# invert y axis
initial_mu = -initial_mu
mu_list = [-i for i in mu_list]
initial_var = -initial_var
var_list = [-i for i in var_list]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)


#torino
#QPDE
gap = -20.91149746860633
shots = 1e4
# initial value
initial_mu = 20.23102405089335 #value of MPS maxdim 2
initial_var = 4.0
png_save_name="TQPDE_vac_Hubbard_4site_torino"
label = "Exact"
prob_list=[[0.2083631, 0.3074153, 0.3603488, 0.2881449, 0.3712898, 0.4025395, 0.5846644, 0.6776161, 0.6536881, 0.6850835, 0.6950026, 0.7302871, 0.7190768, 0.6605381, 0.6826701, 0.6739855, 0.6079248, 0.518544, 0.4430299, 0.400741, 0.3274342], [0.0971003, 0.1224636, 0.1282834, 0.154451, 0.1429072, 0.196971, 0.2136971, 0.3426636, 0.2761884, 0.3122756, 0.3404511, 0.3666014, 0.363511, 0.3901898, 0.4336885, 0.4003843, 0.406175, 0.3782204, 0.3569452, 0.3225782, 0.3254846], [0.0179504, 0.0252727, 0.0347123, 0.0387289, 0.0432336, 0.0437631, 0.0507186, 0.056125, 0.0545398, 0.0661256, 0.0775044, 0.0974523, 0.1107486, 0.1002922, 0.1097289, 0.1133101, 0.1024284, 0.1114353, 0.1101039, 0.0923248, 0.0689243], [0.016674, 0.016758, 0.0174583, 0.0218758, 0.0177665, 0.0259649, 0.0181507, 0.0233973, 0.0203473, 0.0238511, 0.0215673, 0.0193111, 0.0194548, 0.0206644, 0.0192161, 0.0160361, 0.0209581, 0.0174969, 0.0219643, 0.0152349, 0.0138814], [0.0061474, 0.0060945, 0.0055353, 0.0057601, 0.0074643, 0.0062091, 0.0061275, 0.0068669, 0.0067741, 0.0050883, 0.0065227, 0.0054914, 0.0055824, 0.002991, 0.0054979, 0.0054485, 0.0068392, 0.0079585, 0.0051084, 0.007252, 0.0074332], [0.0064, 0.0050301, 0.0051377, 0.0076267, 0.0066233, 0.006064, 0.0052617, 0.0068578, 0.0068706, 0.00751, 0.0058917, 0.0050613, 0.0067051, 0.0070617, 0.0071139, 0.006658, 0.0075936, 0.0070786, 0.0066939, 0.0071161, 0.0071005], [0.0045352, 0.0037705, 0.0057188, 0.0053064, 0.0049708, 0.005443, 0.0071535, 0.0061104, 0.0040545, 0.0044006, 0.0052266, 0.0060856, 0.0059299, 0.0079652, 0.0062463, 0.0050344, 0.0047629, 0.0062584, 0.0065374, 0.0049788, 0.0056523], [0.0024781, 0.001928, 0.0022759, 0.0022382, 0.0019541, 0.0020419, 0.0023342, 0.0015993, 0.0017796, 0.0022227, 0.0018177, 0.0029662, 0.0018386, 0.0021196, 0.0015782, 0.0024767, 0.0015902, 0.0019026, 0.0018578, 0.0016972, 0.0014471]]
mu_list=[20.408389, 20.828836, 21.2631, 21.228158, 21.253996, 21.346696, 21.406551, 21.326236]
var_list=[2.659256, 1.699378, 0.83792, 0.497996, 0.497957, 0.478805, 0.298355, 0.196782]

# invert y axis
initial_mu = -initial_mu
mu_list = [-i for i in mu_list]
initial_var = -initial_var
var_list = [-i for i in var_list]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)


#21 qubit wo qctrl Hubbard
#QPDE
gap = 0.12579074827220627 #MPS
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
png_save_name="TQPDE_woqctrl_Hubbard_10site_torino"
label = "MPS"
prob_list=[[0.0001, 0.0001, 0.0004, 0.0007, 0.0003, 0.0006, 0.0004, 0.0001, 0.0005, 0.0011, 0.0003, 0.0007, 0.0006, 0.0005, 0.0007, 0.0003, 0.0005, 0.0002, 0.0002, 0.0005, 0.0005], [0.0, 0.0001, 0.0001, 0.0002, 0.0001, 0.0003, 0.0001, 0.0001, 0.0003, 0.0002, 0.0, 0.0001, 0.0005, 0.0001, 0.0003, 0.0002, 0.0002, 0.0, 0.0002, 0.0002, 0.0001], [0.0001, 0.0, 0.0, 0.0001, 0.0002, 0.0001, 0.0002, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0001, 0.0002, 0.0, 0.0, 0.0001, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
mu_list=[0.083222, 0.237324, -0.148802, -0.495177, -0.495177]
var_list=[2.867904, 1.892386, 1.378213, 0.659874, 0.329937]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#octatetraene
#aer
gap = 0.11255404976208183
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
png_save_name="TQPDE_octatetraene_aer"
label = "Exact"
prob_list=[[0.2431, 0.3248, 0.4233, 0.5193, 0.6182, 0.7085, 0.7865, 0.8567, 0.9139, 0.9542, 0.9699, 0.9695, 0.9544, 0.9135, 0.8605, 0.7971, 0.7054, 0.6197, 0.5215, 0.4236, 0.3363], [0.228, 0.3117, 0.408, 0.4911, 0.5908, 0.6811, 0.7576, 0.8273, 0.8736, 0.9138, 0.9316, 0.9335, 0.9115, 0.8757, 0.8223, 0.7545, 0.6736, 0.5918, 0.4946, 0.3869, 0.3162], [0.2683, 0.3394, 0.4218, 0.5001, 0.5741, 0.6412, 0.7044, 0.7402, 0.7874, 0.8196, 0.8325, 0.8255, 0.8203, 0.7979, 0.7478, 0.7, 0.6435, 0.57, 0.4995, 0.4191, 0.3544], [0.1983, 0.2499, 0.3051, 0.3689, 0.4199, 0.4636, 0.5019, 0.5382, 0.5675, 0.588, 0.6031, 0.5904, 0.5835, 0.5627, 0.5422, 0.4975, 0.4532, 0.414, 0.3654, 0.3044, 0.2572], [0.1853, 0.2183, 0.244, 0.28, 0.3026, 0.3295, 0.3466, 0.361, 0.3625, 0.3642, 0.3561, 0.3594, 0.3386, 0.3189, 0.2979, 0.2708, 0.2404, 0.2069, 0.1729, 0.142, 0.1145], [0.331, 0.336, 0.3444, 0.3487, 0.3476, 0.334, 0.3181, 0.3006, 0.2777, 0.2567, 0.2204, 0.185, 0.1545, 0.1317, 0.0956, 0.0718, 0.0504, 0.0266, 0.0139, 0.004, 0.0007]]
mu_list=[0.078996, 0.139752, 0.181674, 0.20216, 0.188807, 0.183711]
var_list=[2.532016, 1.311129, 0.534296, 0.119268, 0.007559, 2.5e-05]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#torino
gap = 0.11255404976208183
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
png_save_name="TQPDE_octatetraene_torino"
label = "Exact"
prob_list=[[0.1699662, 0.1765608, 0.270613, 0.2904446, 0.2901531, 0.3753951, 0.3947441, 0.4836478, 0.4814621, 0.5540456, 0.5529957, 0.4706119, 0.3880919, 0.5068643, 0.4338352, 0.4013504, 0.4173709, 0.2782288, 0.3951185, 0.274708, 0.1884625], [0.1071138, 0.1234475, 0.1165092, 0.2269891, 0.2130931, 0.2382139, 0.2320048, 0.2437298, 0.3518808, 0.1995245, 0.2259232, 0.2435354, 0.2083848, 0.2420484, 0.2212895, 0.2405086, 0.142327, 0.1357875, 0.1050503, 0.0541684, 0.0571695], [0.0004798, 0.0002576, 0.0006636, 0.0002069, 2.53e-05, 0.000655, 0.000394, 0.0008709, 0.0009168, 0.000274, 0.0, 0.0004651, 0.0, 0.0002573, 0.0008628, 0.0006916, 0.0006969, 0.0012968, 0.0003975, 0.0002704, 1.77e-05],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003297, 0.0001315, 0.0001094, 0.0, 0.0, 0.0, 0.0]]
mu_list=[0.081096, -0.049315, 0.048381, 0.525009]
var_list=[2.701453, 1.488544, 1.108881, 0.002588]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#hexatriene
#local
#aer
gap = 0.12461581875911065
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
png_save_name="TQPDE_hexatriene_local_aer"
label = "Exact"
prob_list=[[0.2358, 0.3253, 0.4205, 0.5188, 0.6194, 0.7008, 0.7892, 0.8639, 0.9212, 0.9592, 0.9822, 0.9829, 0.9659, 0.9309, 0.8732, 0.8102, 0.7269, 0.6308, 0.5435, 0.4336, 0.3282], [0.2261, 0.3132, 0.3999, 0.5016, 0.5973, 0.6879, 0.7762, 0.8481, 0.9036, 0.9398, 0.9571, 0.9581, 0.9452, 0.9041, 0.8516, 0.7904, 0.7103, 0.6151, 0.5145, 0.4147, 0.3162], [0.2918, 0.3703, 0.4489, 0.532, 0.6136, 0.6891, 0.7616, 0.8182, 0.856, 0.8895, 0.9034, 0.9002, 0.8917, 0.8646, 0.8259, 0.7675, 0.7051, 0.627, 0.562, 0.4843, 0.3861], [0.2499, 0.3286, 0.3831, 0.4597, 0.5171, 0.5846, 0.6336, 0.6779, 0.726, 0.7446, 0.761, 0.7703, 0.7619, 0.7361, 0.7062, 0.6546, 0.6058, 0.5371, 0.4741, 0.4106, 0.342], [0.3217, 0.39, 0.456, 0.5231, 0.5731, 0.6273, 0.6734, 0.7004, 0.7374, 0.7538, 0.7519, 0.7364, 0.7147, 0.6841, 0.6394, 0.5822, 0.5191, 0.4617, 0.3955, 0.3242, 0.2637], [0.3014, 0.3683, 0.4353, 0.495, 0.5378, 0.5949, 0.6306, 0.6624, 0.6904, 0.7005, 0.6971, 0.6856, 0.6659, 0.6323, 0.5866, 0.533, 0.4763, 0.4177, 0.3656, 0.2999, 0.2417]]
mu_list=[0.090469, 0.163218, 0.213106, 0.241968, 0.236827, 0.236406]
var_list=[2.525734, 1.300148, 0.530564, 0.118731, 0.007243, 2.9e-05]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#torino
gap = 0.12461581875911065
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
png_save_name="TQPDE_hexatriene_local_torino"
label = "Exact"
prob_list=[[0.1840995, 0.1687645, 0.2774065, 0.3455126, 0.3055877, 0.4168849, 0.4936611, 0.5138537, 0.5063334, 0.6319118, 0.4993982, 0.5491751, 0.5413981, 0.5564775, 0.4744838, 0.5519523, 0.5281508, 0.396721, 0.38593, 0.3428951, 0.2987846], [0.1135985, 0.1786057, 0.1815126, 0.1970329, 0.2422779, 0.2478932, 0.246878, 0.2992873, 0.2978613, 0.3011156, 0.2969624, 0.2846575, 0.2485502, 0.2504439, 0.1773339, 0.1844897, 0.1917684, 0.1327346, 0.1109384, 0.0603697, 0.0637773], [0.0047644, 0.0042697, 0.0065999, 0.0063014, 0.0058332, 0.0095754, 0.0043785, 0.0060241, 0.0066103, 0.0057608, 0.005883, 0.0066275, 0.0057803, 0.0058371, 0.0047495, 0.0034332, 0.0054764, 0.0050048, 0.0056696, 0.0023208, 0.0028912], [0.0007489, 0.0012846, 0.0004878, 0.0004019, 0.0007362, 0.0015285, 0.0002634, 0.0006854, 0.0011078, 0.0017265, 0.0003711, 0.000971, 0.0009472, 0.0007108, 0.0012997, 0.0013509, 0.000902, 0.0002042, 0.0005315, 0.0003774, 0.0011901]]
mu_list=[0.156604, -0.021259, -0.172611, -0.185062]
var_list=[2.780213, 1.516726, 0.9672, 0.590848]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)


#HF
#aer
gap = 0.12461581875911065
shots = 1e4
# initial value
initial_mu = 0.01
initial_var = 4.0
png_save_name="TQPDE_hexatriene_HF_aer"
label = "Exact"
prob_list=[[0.3289, 0.4278, 0.5192, 0.6212, 0.7028, 0.7994, 0.8674, 0.9279, 0.9689, 0.9893, 0.9921, 0.9771, 0.9429, 0.8854, 0.8239, 0.7354, 0.6484, 0.5417, 0.4502, 0.3525, 0.26], [0.3077, 0.401, 0.5032, 0.5972, 0.6956, 0.7829, 0.8547, 0.9141, 0.9579, 0.9836, 0.9852, 0.9687, 0.9302, 0.8829, 0.809, 0.7301, 0.6325, 0.5369, 0.4333, 0.3394, 0.243], [0.3879, 0.4726, 0.5536, 0.6428, 0.721, 0.8019, 0.8611, 0.909, 0.9402, 0.958, 0.9637, 0.9478, 0.9244, 0.8811, 0.8225, 0.7521, 0.6808, 0.6092, 0.5079, 0.4217, 0.3262], [0.3212, 0.3966, 0.4789, 0.5527, 0.6111, 0.6551, 0.7166, 0.7578, 0.7944, 0.7998, 0.805, 0.7836, 0.7688, 0.7305, 0.6861, 0.6374, 0.5789, 0.5081, 0.4384, 0.3659, 0.2942], [0.2486, 0.27, 0.2816, 0.2935, 0.3041, 0.3153, 0.3149, 0.3164, 0.3089, 0.3024, 0.2911, 0.2757, 0.2531, 0.2304, 0.2143, 0.1896, 0.175, 0.1504, 0.1358, 0.1135, 0.1008], [0.1101, 0.1243, 0.1201, 0.1197, 0.1284, 0.1275, 0.1283, 0.1241, 0.1183, 0.1033, 0.0912, 0.0805, 0.0826, 0.0662, 0.0546, 0.0483, 0.0382, 0.0292, 0.0234, 0.0168, 0.0135]]
mu_list=[-0.044914, -0.086652, -0.109717, -0.120982, -0.28326, -0.299968]
var_list=[2.537642, 1.305782, 0.53216, 0.120261, 0.010628, 6.3e-05]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#torino
gap = 0.12461581875911065
shots = 1e4
# initial value
initial_mu = 0.0
initial_var = 4.0
png_save_name="TQPDE_hexatriene_HF_torino"
label = "Exact"
prob_list=[[0.1491075, 0.2081265, 0.2661671, 0.2592718, 0.2687953, 0.2486937, 0.2820415, 0.3062557, 0.3183364, 0.3598127, 0.317408, 0.3424784, 0.2519842, 0.2590792, 0.1997735, 0.1461711, 0.1730439, 0.1092808, 0.1010622, 0.0577281, 0.0622816], [0.0313178, 0.0172643, 0.0411078, 0.0402182, 0.0360779, 0.064165, 0.0655356, 0.0701352, 0.0715867, 0.080519, 0.0831992, 0.0745583, 0.0726454, 0.0760158, 0.0697806, 0.0665752, 0.0410077, 0.0343654, 0.0375069, 0.0184976, 0.0331225], [0.0011575, 0.0, 0.0002884, 0.0001521, 0.0, 0.0005514, 0.0001436, 0.0002069, 0.0002377, 0.0004875, 5.87e-05, 0.0003335, 0.0006396, 0.0003764, 0.0001266, 0.0002981, 0.0, 0.0004548, 0.0, 0.0005314, 0.0002806]]
mu_list=[-0.315764, -0.306504, -0.533772]
var_list=[2.533189, 1.237583, 1.236897]

gap_list.append(gap)
shots_list.append(shots)
initial_mu_list.append(initial_mu)
initial_var_list.append(initial_var)
png_save_name_list.append(png_save_name)
prob_list_list.append(prob_list)
mu_list_list.append(mu_list)
var_list_list.append(var_list)
label_list.append(label)

#============================================
for initial_mu, initial_var, gap, label, png_save_name, prob_list, mu_list, var_list in zip(initial_mu_list, initial_var_list, gap_list, label_list, png_save_name_list, prob_list_list, mu_list_list, var_list_list):
    
    print(png_save_name)
    if len(prob_list) != len(mu_list) or len(mu_list) != len(var_list):
        print(len(prob_list))
        print(len(mu_list))
        print(len(var_list))
        raise Exception("length prob_list, mu_list, and var_list are different")

    # num iteration
    iterations = np.arange(0, len(mu_list) + 1)
    mu = [initial_mu] + mu_list
    sigma = [np.sqrt(np.abs(initial_var))] + [np.sqrt(np.abs(var)) for var in var_list]
    var = [initial_var] + var_list

    # List to store sampled values and probabilities for each iteration
    sampled_values = []
    probabilities = []

    # Sampling and probability setting
    for i, probs in enumerate(prob_list):
        samples = np.linspace(mu[i] - var[i], mu[i] + var[i], len(probs))
        sampled_values.append(samples)
        probabilities.append(probs)

    # Create a list of all probabilities and obtain maximum and minimum values
    all_probs = [prob for sublist in probabilities for prob in sublist]
    min_prob = min(all_probs)
    max_prob = max(all_probs)

    # create plot
    plt.figure(figsize=(10, 6))

    # plot mu sigma
    plt.errorbar(iterations, mu, yerr=sigma, fmt='o', color='black', ecolor='black', capsize=5)

    plt.hlines(gap,xmin=-1,xmax=len(var_list)+1,linestyles="--",colors="black",label=label)

    #normalize for each iter~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot the probability distributions as left and right expansions
    for i, (samples, probs) in enumerate(zip(sampled_values, probabilities)):
        y = np.array(samples)
        probs = np.array(probs)
        if max(probs) == 0.0:
            probs_normalized = probs
        else:
            probs_normalized = probs * 0.8 / max(probs)
        plt.fill_betweenx(y, i + 1, i + 1 + probs_normalized, color='red', alpha=0.5)
        plt.text(i+1, max([max(samples) for samples in sampled_values])+0.65, f"{max(probs):.4f}", 
                horizontalalignment='left', verticalalignment='center', fontsize=13, color='black')

    png_save_name += "_normalize_each_iter"
    plt.text(0.8, max([max(samples) for samples in sampled_values])+0.65, f"Max prob.", 
            horizontalalignment='right', verticalalignment='center', fontsize=13, color='black')
    #end normalize for each iter~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.xlabel('Iteration')

    if png_save_name.count("butadiene") == 1:
        plt.ylabel('Energy (Hartree)')
    else:
        plt.ylabel('Energy')
    plt.xticks(iterations)  # Display iterations as integers
    plt.xlim(-0.1, len(var_list)+0.9)

    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
    plt.legend(loc="lower right")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plt.show()
    plt.savefig(dir_QPDE_path+"/picture/"+png_save_name+".png")