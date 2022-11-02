# Seq2seq


## TODO
- [] add inference notebook
- [x] add letter removing task

## Train
```
python train.py --workdir ./logs --task select_letters_by_mask --num_train_steps 1000 --max_len_query_digit 10
```