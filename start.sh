#!/usr/bin/env sh


##################################################################
function create_leveldb(){
BINPATH=$1
IMGPATH=$2      #where is our images
LABELFILE=$3    #image names and label index
DSTPATH=$4      #where to put the leveldb file
RESIZE_HEIGHT=$5
RESIZE_WIDTH=$6

GLOG_logtostderr=1 $BINPATH/convert_imageset \
        -resize_height=$RESIZE_HEIGHT \
        -resize_width=$RESIZE_WIDTH \
		-encode_type='png' \
		-encoded=true \
		-shuffle=true \
        $IMGPATH \
        $LABELFILE \
        $DSTPATH

}

function train_init()
{
	log=$1
	GLOG_logtostderr=1 $BINPATH/caffe train \
	--solver=imagenet_solver.prototxt \
	--weights=nin_imagenet.caffemodel  \
	 2>&1 | tee $log
}

##################################################################


##################################################################

##################################################################

#caffe�Ķ�����Ŀ¼
BINPATH=/data/caffe/build/tools/
#��Ҫ����ѵ�������б�,ѵ�����Ϻ���֤���϶�������,�����Զ�������ѡ
#src=../data/cnn2.label2

#��ʼ��ѡ
echo "Start to choose train and valid set"
#train_ratio=95

#��ݱ����������,Ȼ������������,��ȡTopK��
#cnt_total=$(cat $src | wc -l)
#cnt_train=`expr $cnt_total \* $train_ratio`
#cnt_train=`expr $cnt_train / 100`
#cnt_valid=`expr $cnt_total - $cnt_train`

#echo "cnt_train = $cnt_train"
#echo "cnt_valid = $cnt_valid"

mkdir models

train_set=train.txt
valid_set=valid.txt

#tmp=/tmp/$$.allalble
#cat $src | shuf > $tmp
#cat $tmp | head -n $cnt_train > $train_set
#cat $tmp | tail -n $cnt_valid > $valid_set

#����leveldb�ļ�,����
echo "Start to create level db files"
RESIZE_HEIGHT=128
RESIZE_WIDTH=128

train_db=train_lmdb
valid_db=valid_lmdb

rm -rf $train_db
#rm -rf $valid_db
create_leveldb  $BINPATH / $train_set $train_db $RESIZE_HEIGHT $RESIZE_WIDTH
#create_leveldb  $BINPATH / $valid_set $valid_db $RESIZE_HEIGHT $RESIZE_WIDTH
wait

#Compute Image Mean
#echo "Start to Compute Image Mean"
#train_mean=mean.binaryproto
#$BINPATH/compute_image_mean $train_db $train_mean


#start
echo "Start to train"
#solver_model=imagenet_solver.prototxt  #solver.prototxt
#$BINPATH/caffe train --solver=$solver_model --weights=nin_imagenet.caffemodel

MODELS_DIR=./models/
LOG_DIR=./train_log/
if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi
if [ ! -d "$MODELS_DIR" ];then
	mkdir $MODELS_DIR
fi

log_file="train_caffe_"`date +%Y%m%d-%H%M`".log"
init_log=${LOG_DIR}${log_file}
(train_init $init_log &)

nohup sh ./watch.sh $init_log &
