# RobotNavigateNLP
OSU 5522 AI project: navigate robots with natural language instructions

Duty | Who | Detail
------------ | ------------- | ------------
Program front-end | Shuang Li  | front-end presentation of three maps, interaction with back-end model 
Process data set | Qi Zhu | processing data into pairs of nl instructions and action sequences, map environment matrices
Code model with TensorFlow | Ziyu Yao | code seq2seq model with environment representation

## matrix
- [x, y, d], d = 0, 90, 180, 270
- x, y表示一个点，d表示方向，0表示右
- [x, y, d] = object(6 * 自前后左右), floor + wall(11 * 前后左右) 
- Update: make a 3d matrix for each map of shape [sizeOfX, sizeOfY, 4]. Each element is a 74d vector(implemented with python list). If (x, y, d) doesn't exist in the map, output a zero vector with the same length.

## function
Update: cancel this function. See "matrix".
