#!/bin/bash

mhm2_install_dir=$(dirname $(dirname $(realpath $0) ) )
if [ -n "$1" -a -d "$1" ] || [ ! -x ${mhm2_install_dir}/bin/ci_asm_qual_test.sh ]
then
  mhm2_install_dir=$(realpath $1)
  shift
fi

refs=arcticsynth-refs.fa
if [ ! -f "$refs" ]; then
    rm -f ${refs}.gz
    wget https://portal.nersc.gov/project/hipmer/MetaHipMer_datasets_12_2019/ArcticSynth/samples/${refs}.gz
    gunzip ${refs}.gz &
fi
reads=arctic_sample_0.fq
if [ ! -f "$reads" ]; then
    rm -f ${reads}.gz
    wget https://portal.nersc.gov/project/hipmer/MetaHipMer_datasets_12_2019/ArcticSynth/samples/${reads}.gz
    gunzip ${reads}.gz &
fi 

wait

wd=`pwd`
test_dir=$wd/test-arctic-sample0
if [[ "$*" != *"--restart"* ]]
then
  rm -rf $test_dir
else
  echo "Restarting in $test_dir"
fi
$mhm2_install_dir/bin/mhm2.py $@ -r $reads -o $test_dir --checkpoint=no --post-asm-align --post-asm-abd
status=$?
if [ $status -ne 0 ]
then
  echo "MHM2 failed! - $status"
  exit $status
fi
$mhm2_install_dir/bin/check_asm_quality.py --asm-dir $test_dir --expected-quals $mhm2_install_dir/share/good-arctic-sample0.txt --refs $wd/$refs 2>&1 \
   | tee $test_dir/check_asm_quality_test.log
status=$?
if [ $status -ne 0 ]
then
  echo "check_asm_quality.py failed! - $status"
  exit $status
fi
