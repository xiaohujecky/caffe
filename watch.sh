#!/usr/bin/env sh

TOOLS=/data/caffe/build/tools
MODELS_DIR=./models/
LOG_DIR=./train_log/
MAX_RESTART_TIME=5

function train_init()
{
	log=$1
	GLOG_logtostderr=1 $TOOLS/caffe train \
	--solver=imagenet_solver.prototxt \
	--weights=nin_imagenet.caffemodel  \
	 2>&1 | tee $log
}

function train_snapshot()
{
	log=$1
	snapshot=$2
	GLOG_logtostderr=1 $TOOLS/caffe train \
	--solver=imagenet_solver.prototxt \
	--snapshot=${snapshot} \
	 2>&1 | tee $log
}

function run_train()
{
	iter_id=$1
	log=$2
	#nohup sh ./train_full.sh >${log_dir}${log} &
	if [ "00"x == "$iter_id"x ];then
		train_init $log
	else
		train_snapshot $log $iter_id
	fi
	echo "${log}"
}

function train_watch()
{
	log=$1
	while :
	do
		#echo "$i"
		sleep 10
		snapshot=`cat $log |grep Snapshotting | grep solverstate$ | tail -n 1 | awk '{print $NF}'`
		val_loss=`cat $log | grep loss | tail -n 1 | awk '{print $NF}'`
		process_id=`cat $log | grep loss | tail -n 1 | awk '{print $3}'`
		iter_cnt=`cat $log | grep loss | tail -n 1 | awk '{print $6}' | sed 's:,::'`
		val_lr=`cat $log | grep lr | tail -n 1 | awk '{print $NF}'`
		echo "$process_id $iter_cnt $val_loss $val_lr $snapshot" >> watch_log
		#cat watch_log | sort | uniq > tmp.$$;mv tmp.$$ watch_log
		if [ $(echo "$val_loss > 20" | bc) -eq 1 ];then
			echo "$process_id $iter_cnt $val_loss $val_lr $snapshot" >> watch_log
			kill $process_id
			echo "$snapshot loss_fail"
			break
		fi
		train_finish=`cat $log | grep caffe.cpp | grep Optimization  | tail -n 1 | awk '{print $NF}'`
		if [ "Done."x == "$train_finish"x ];then
			echo "exit"
			exit 0
		fi
	done
}


function watch()
{

#init_log=$1
cnt=1
if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi
if [ ! -d "$MODELS_DIR" ];then
	mkdir $MODELS_DIR
fi
#log_file="train_caffe_"`date +%Y%m%d-%H%M`".log"
#init_log=${LOG_DIR}${log_file}
#train_init $init_log
init_log="./train_log/train_caffe_20160619-2347.log"

info=$(train_watch $init_log)
while :
do
	info1=`echo $info | grep "loss_fail"`
	snapshot=`echo $info1 | awk '{print $1}'`
	echo "loss fail, restart from $snapshot" >> watch_log
	log_file="train_caffe_"`date +%Y%m%d-%H%M`".log"
	log=${LOG_DIR}${log_file}
	(run_train $snapshot $log &)
	info=$(train_watch $log)
	((cnt++))
	if [ $cnt -gt $MAX_RESTART_TIME ]
	then
		echo "restart for $cnt times, exit."
		exit 0
	fi
done
}

mailSend()
{
        mailContent="watch exit: for response time over 1 day!"
        #echo $mailContent | mail -s "Test TimeOut" i-ruanxiaohu@alibaba-inc.com
		echo $mailContent
}

timeout()
{
        waitfor=180000
        command=$*
        $command &
        commandpid=$!

        ( sleep $waitfor ; kill -9 $commandpid  > /dev/null 2>&1 && mailSend ) &

        watchdog=$!
        sleeppid=$PPID
        wait $commandpid > /dev/null 2>&1

        kill $sleeppid > /dev/null 2>&1
}

timeout watch
