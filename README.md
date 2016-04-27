# RobotNavigateNLP
OSU 5522 AI project: navigate robots with natural language instructions

Duty | Who | Detail
------------ | ------------- | ------------
Program front-end | Shuang Li  | front-end presentation of three maps, interaction with back-end model 
Process data set | Qi Zhu | processing data into pairs of nl instructions and action sequences, map environment matrices
Code model with TensorFlow | Ziyu Yao | code seq2seq model with environment representation

## Dataset
### Original dataset
- The original single-sentence MARCO dataset from <http://www.cs.utexas.edu/users/ml/clamp/navigation/downloads/LearningNavigationInstructions/>.
- Original example in map "Grid":

Instruction | Action sequence 
:------------: | :-------------: 
face the octagon carpet | (3, 5, -1), (3, 5, 180)  

### Data preparation
- Use 0 ~ 4 to represent action "no action", "move forward 1 step", "turn right", "turn left", "turn back" in action sequence. Intial directions are set to 90 degree.
-  (x, y, d) is the position (x,y) in maps with facial direction d, where d in {0, 1, 2, 3} corresponds to degree direction {0, 90, 180, 270} respectively.
-  Three maps are indexed with number, where 0 = "Grid", 1 = "Jelly", 2 = "L".
- Processed example:

Instruction | Action sequence | Action positions | Map
:------------: | :-----------: | :----:| :---:
face the octagon carpet | 3 | (3, 5, 1), (3, 5, 2) | map0

## Map representation
- see seq2seq/map.py
- [x, y, d], d = 0, 90, 180, 270
- x, y表示一个点，d表示方向，0表示右
- [x, y, d] = object(6 * 自前后左右), floor + wall(11 * 前后左右) 
- Update: make a 3d matrix for each map of shape [sizeOfX, sizeOfY, 4]. Each element is a 74d vector(implemented with python list). If (x, y, d) doesn't exist in the map, output a zero vector with the same length.

### How to use matrix

```
cd matrix
python
// in python 
import map // at the directory of matrix
map.map_grid
map.map_jelly
map.map_one
```


<!--## function

- input: [x0, y0, d0], operation (0 = no action, 1 = go forward, 2 = turn right, 3 = turn left, 4 = turn back)
- output: [x1, y1, d1]
- if there is no way ahead, return null
-->

## Train and decode
### Train
```
rm ../data/data0/checkpoint/* # clean existing checkpoint records
python seq2seq_run.py --size --learning_rate --keep_prob --epoch --num_layer --batch_size --data_dir="../data/data0"
```

### Decode
#### Interactive decode
```
python seq2seq_run.py --size --learning_rate --keep_prob --epoch --num_layer --data_dir="../data/data0" --inter_decode=True --batch_size=1
```
#### Decode for testing
```
python seq2seq_run.py --size --learning_rate --keep_prob --epoch --num_layer --data_dir="../data/data0" --decode=True --decode_dev=True(or --decode_test=True) --batch_size=1 --inter_decode_sent="string_of_instruction" --inter_decode_position="[[1,7,2]]" --inter_decode_map=0/1/2
```
