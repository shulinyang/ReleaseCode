#!/bin/bash
# author: Daneel
# domain_type=(3 2 4)

domain_list=(4 3 2 1 0)
# domain_list=(2 3)
index=0
domain_name=${10}
domain_index=$9
while (( $index < ${#domain_list[*]} ))
do
   # log_txt="$1_$2_$6_3PCA_dt_${domain_type[$index]}_tt_1_"$(date "+%Y%m%d%H%M%S")""
   log_txt="0_$1_$2_$6_${domain_name}_${domain_list[$index]}_$8_"$(date "+%Y%m%d%H%M%S")""
   echo ${index}
   `CUDA_VISIBLE_DEVICES=$3 python main.py -ds $5 -nt $6 -dt ${domain_index} -tt 1 -ss remote -pt $1 -gpu $4 -ttr 0.8 -mn $2 -tl $7 -receiver 0 1 2 3 4 5 -pca 0 1 2 -pa 10 -gesture 1 2 3 4 5 6 -user 1 2 3 4 5 6 -${domain_name} ${domain_list[$index]} > experiment/$1/$8/${log_txt}`
   # `CUDA_VISIBLE_DEVICES=$3 python main.py -ds $5 -nt $6 -dt 2 -tt 1 -ss remote -pt $1 -gpu $4 -ttr 0.8 -mn $2 -tl $7 -receiver 0 1 2 3 4 5 -pca 0 1 2 -pa 30 -gesture 1 2 3 4 5 6 -user 1 2 3 4 5 6 -pd ${domain_list[$index]} > logs/${log_txt}`
   # `CUDA_VISIBLE_DEVICES=$3 python main.py -ds $5 -nt $6 -dt ${domain_type[$index]} -tt 1 -pt $1 -tl $7 -mn $2 -receiver 0 1 2 3 4 5 -pca 0 1 2 -gpu $4 -ss remote -gesture 1 2 3 4 5 6 -user 1 2 3 4 5 6 -grr $8 > experiment/$1/id/${log_txt}`
   size=`stat -c "%s" experiment/$1/$8/${log_txt}`
   echo ${size}
   if (( ${size} > 2000 ))
   then
      echo "logging success!"
      index=`expr $index + 1`
   else
      `rm experiment/$1/$8/${log_txt}`
      echo "log error, do it again!"
   fi
done
