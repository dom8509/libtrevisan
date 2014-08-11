#!/bin/bash

rotateCursor() {
  case $toggle
  in
    1)
      echo -n $1" \ "
      echo -ne "\r"
      toggle="2"
    ;;

    2)
      echo -n $1" | "
      echo -ne "\r"
      toggle="3"
    ;;

    3)
      echo -n $1" / "
      echo -ne "\r"
      toggle="4"
    ;;

    *)
      echo -n $1" - "
      echo -ne "\r"
      toggle="1"
    ;;
  esac
}

mkdir tmp 2>/dev/null
rm tmp/* 2>/dev/null


######################################################
#invoke rsh extractor
let breakout=1
trap "let breakout=0" USR1

#background process, raise USR1 signal
export this_pid=$$

text="Invoking RSH Extractor... "
../extractor -n 1000000 -m 5000 -x rsh --extRand-file tmp/rsh 1>/dev/null 2>&1 && kill -USR1 $this_pid 2>/dev/null &

while [[ $breakout -eq 1 ]]
do 
  rotateCursor "$text"
  sleep .03
done

echo -n "$text"
echo -e "\e[00;32mfinished\e[00m"

######################################################
#invoke rsh_cuda extractor
let breakout=1
trap "let breakout=0" USR1

#background process, raise USR1 signal
export this_pid=$$

text="Invoking RSH_CUDA Extractor... "
../extractor -n 1000000 -m 5000 -x rsh_cuda --extRand-file tmp/rsh_cuda 1>/dev/null 2>&1 && kill -USR1 $this_pid 2>/dev/null &

while [[ $breakout -eq 1 ]]
do 
  rotateCursor "$text"
  sleep .03
done

echo -n "$text"
echo -e "\e[00;32mfinished\e[00m"


echo -n 'Comparing Extracted Randomness... '

md5rsh=($(md5sum tmp/rsh))
md5rsh_cuda=($(md5sum tmp/rsh_cuda))

#echo $md5rsh
#echo $md5rsh_cuda

if [ $md5rsh != $md5rsh_cuda ]; then
	echo -e "\e[00;31mmd5 mismatch!\e[00m"
else
	echo -e "\e[00;31mRED\e[00m"
fi
