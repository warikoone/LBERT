for i in {1..10}
do
	for m in {1..1}
	do
		echo -e '\n************DataSet '$i'************\t************TrainingFold '$m'************ \n';
		python run_re.py \
			--task_name=cisp \
			--do_train=true \
			--do_eval=true \
			--do_predict=true \
			--vocab_file=$LBERTDIR/L-Bert/src/com/prj/bundle/resource/biobert_v1.0_pubmed/vocab.txt \
			--bert_config_file=$LBERTDIR/L-Bert/src/com/prj/bundle/resource/biobert_v1.0_pubmed/bert_config.json \
			--init_checkpoint=$LBERTDIR/ValidationSet_Generation/src/com/prj/bundle/output_19.0/model.ckpt-8971 \
			--feature_dir=$LBERTDIR/SInD/src/com/prj/bundle/processed \
			--layer_def=1,1,1,1 \
			--max_seq_length=128 \
			--train_batch_size=32 \
			--learning_rate=2e-5 \
			--num_train_epochs=3.0 \
			--kernel_size=3 \
			--stride_size=1 \
			--do_lower_case=false \
			--data_dir=$LBERTDIR/ValidationSet_Generation/src/com/prj/bundle/processed/$i \
			--output_dir=$LBERTDIR/ValidationSet_Generation/src/com/prj/bundle/output \
			--test_fold=$m;

		python ../biocodes/re_eval.py --task=chemprot --output_path=$LBERTDIR/ValidationSet_Generation/src/com/prj/bundle/processed/$i/test_results$m.tsv --answer_path=$LBERTDIR/ValidationSet_Generation/src/com/prj/bundle/processed/$i/test.tsv;
		
		if [ $i == 0 -a $m == 0 ]
		then
			continue
		else
			echo -e '\nRemoving the Model file ' $m '\n';
			rm -R $LBERTDIR/ValidationSet_Generation/src/com/prj/bundle/output;
		fi
	done
done
