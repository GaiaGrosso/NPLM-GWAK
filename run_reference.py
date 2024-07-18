import os, json, argparse, glob, time, datetime
import numpy as np
import os.path

OUTPUT_DIRECTORY = './dictionary-learning/out/'
def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

# configuration dictionary
config_json = {
    "N_Ref"   : 600,
    "N_Data"   : 60,
    "output_directory": OUTPUT_DIRECTORY,
    "lumi_frac": 1,
    "epochs": 1000,
    "sub_epochs": 100,
    "patience": 100,
    "plt_patience": 500,
    "width_init": 4.0, #[0.1, 0.3, 0.7, 1.4, 3.0]
    "number_centroids": 10,#1000,
    "coeffs_reg_lambda":  0.00000000001,
    "widths_reg_lambda":  0,
    "entropy_reg_lambda": 1.,
    "resolution_scale": [0.01],
    "resolution_const": [0],
    "coeffs_clip": 1000000,
    "train_coeffs": True,
    "train_widths": False,
    "train_centroids": True,
}

# training specs
ID = "M%i_Lcoeffs%s_Lwidths%s_Lentropy%s"%(config_json["number_centroids"],
                                            str(config_json["coeffs_reg_lambda"]),
                                            str(config_json["widths_reg_lambda"]),
                                            str(config_json["entropy_reg_lambda"]))
ID += '_epochs%i'%(config_json['epochs'])
ID += '_subepochs%i'%(config_json['sub_epochs'])
ID += '_width%s'%(str(config_json["width_init"]))
if "gather_after" in list(config_json.keys()):
    ID+='_gathergrad%i'%(config_json["gather_after"])
if config_json["train_coeffs"]==True:
    ID += '_train-coeffs'
if config_json["train_widths"]==True:
    ID += '_train-widths'
if config_json["train_centroids"]==True:
    ID += '_train-centroids'

# problem specs
ID += '_NR'+str(config_json["N_Ref"])+'_ND'+str(config_json["N_Data"])
ID += '_background'
res_bound_string = '/res-scale'
for r in config_json["resolution_scale"]:
    res_bound_string +="_%s"%(str(r))
ID += res_bound_string

coeffs_clip_string = ''
if "coeffs_clip" in list(config_json.keys()):
    coeffs_clip_string+='_coeffs_clip%s'%(config_json["coeffs_clip"])
ID+=coeffs_clip_string
#### launch python script ###########################
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", required=True)
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default=100)
    parser.add_argument('-s', '--firstseed', type=int, help="first seed for toys (if specified the the toys are launched with deterministic seed incresing of one unit)", required=False, default=0)
    args     = parser.parse_args()
    ntoys    = args.toys
    pyscript = args.pyscript
    firstseed= args.firstseed
    config_json['pyscript'] = pyscript
    
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+pyscript_str+'/'+ID
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
    
    json_path = create_config_file(config_json, config_json["output_directory"])

    if args.local:
        if firstseed<0:
            seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            os.system("python %s/%s -j %s -s %i"%(os.getcwd(), pyscript, json_path, seed))
        else:
            os.system("python %s/%s -j %s -s %i"%(os.getcwd(), pyscript, json_path, firstseed))
    else:
        label = "logs"
        os.system("mkdir %s" %label)
        for i in range(ntoys):
            if firstseed>=0:
                seed=i
                seed+=firstseed
            else:
                seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
            script_sbatch = open("%s/submit_%i.sh" %(label, seed) , 'w')
            script_sbatch.write("#!/bin/bash\n")
            script_sbatch.write("#SBATCH -c 1\n")
            script_sbatch.write("#SBATCH --gpus 1\n")
            script_sbatch.write("#SBATCH -t 0-04:00\n")
            script_sbatch.write("#SBATCH -p gpu\n")
            #script_sbatch.write("#SBATCH -p serial_re\n")
            script_sbatch.write("#SBATCH --mem=5000\n")
            script_sbatch.write("#SBATCH -o ./logs/%s"%(pyscript_str)+"_%j.out\n")
            script_sbatch.write("#SBATCH -e ./logs/%s"%(pyscript_str)+"_%j.err\n")
            script_sbatch.write("\n")
            script_sbatch.write("module load python/3.10.9-fasrc01\n")
            script_sbatch.write("module load cuda/11.8.0-fasrc01\n")
            script_sbatch.write("\n")
            script_sbatch.write("python %s/%s -j %s -s %i\n"%(os.getcwd(), pyscript, json_path, seed))
            script_sbatch.close()
            os.system("chmod a+x %s/submit_%i.sh" %(label, seed))
            os.system("sbatch %s/submit_%i.sh"%(label, seed) )

