#!/bin/bash
WORK_DIR=$PWD/observer_results
NPROCS=2
grids=( 64)
crs=( 1)
alphas=( 0.8)
gains=( 5)
seeds=($(seq 123 1 123))
nu=0.006
nustring=$(echo $nu | sed 's/0.//')


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
    for l in "${gains[@]}"
    do
	for s in "${seeds[@]}"
	do
	    for a in "${alphas[@]}"
	    do
	        for c in "${crs[@]}"
		    do
		        JOB_NAME="g${g}_a${a}_cr${c}_s${s}_l${l}_nu0p${nustring}"
			FOLDER=${WORK_DIR}/jaxcfd_noise_grid${g}_alpha${a}_cr${c}_s${s}_l${l}_nu0p${nustring}
			if [ -d $FOLDER ]; then
				echo Deleting existing ${FOLDER} directory
				rm -rf $FOLDER
			else
				echo Creating ${FOLDER} directory
			fi
			mkdir $FOLDER
			cp -rf NoiseLuenbergerTemplate_mpi.py $FOLDER/NoiseLuenberger_mpi.py
                        cd $FOLDER
			sed -i '' "s/gridflag/$g/" NoiseLuenberger_mpi.py
			sed -i '' "s/gainflag/$l/" NoiseLuenberger_mpi.py
			sed -i '' "s/crflag/$c/" NoiseLuenberger_mpi.py
			sed -i '' "s/alphaflag/$a/" NoiseLuenberger_mpi.py
			sed -i '' "s/seedflag/$s/" NoiseLuenberger_mpi.py
			sed -i '' "s/nuflag/$nu/" NoiseLuenberger_mpi.py

			echo Running in $FOLDER
			mpirun -n ${NPROCS} python ../../run_oasis.py problem=NoiseLuenberger_mpi
			cd ..

				done

			done
		done
	done
done
