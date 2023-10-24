#!/bin/bash
WORK_DIR=$PWD/data
grids=( 64)
seeds=($(seq 123 1 123))
nus=( 0.006)
gpu=false
solver=fenics
NPROCS=2

## provide a path to Oasis folder;
## here we are assuming that it is on the same level as this script
cat << EOF > run_oasis.py
#!/usr/bin/python3 -tt
import sys
def main():
	sys.path.insert(0, '../../Oasis/oasis')
	import NSfracStep

if __name__ == '__main__':
	main()

EOF
for g in "${grids[@]}"
do
    for nu in "${nus[@]}"
    do
	for s in "${seeds[@]}"
	do
		FOLDER=${WORK_DIR}/r${s}_g${g}by${g}_nu${nu}
		if [ $solver = "jax-cfd" ]; then
			if [ -d $FOLDER ]; then
				echo Deleting existing ${FOLDER} directory
				rm -rf $FOLDER
			else
				echo Creating ${FOLDER} directory
			fi
			mkdir $FOLDER
			echo Generating Kolmogorov flow results
			echo random seed ${s}, grid ${g} x ${g}, viscosity ${nu} in ${FOLDER}
			if $gpu
			then
				CUDA_VISIBLE_DEVICES=0 python kolmogorov_flow.py ${s} ${g} ${nu} ${FOLDER}
			else
				python kolmogorov_flow.py ${s} ${g} ${nu} ${FOLDER}
			fi
		elif [ $solver = "fenics" ]; then
			if [ -d $FOLDER ]; then
				echo Deleting existing ${FOLDER} directory
				rm -rf $FOLDER
			else
				echo Creating ${FOLDER} directory
			fi
			mkdir $FOLDER
			cp -rf NoiseForwardTemplate_mpi.py $FOLDER/NoiseForward_mpi.py
			cd $FOLDER
			sed -i '' "s/gridflag/$g/" NoiseForward_mpi.py
			sed -i '' "s/nuflag/$nu/" NoiseForward_mpi.py
			sed -i '' "s/seedflag/$s/" NoiseForward_mpi.py
			echo Running in $FOLDER
			mpirun -n ${NPROCS} python ../../run_oasis.py problem=NoiseForward_mpi
		else
			echo There is no such solver!
		fi
		done
	done
done
