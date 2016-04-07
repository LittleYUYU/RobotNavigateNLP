# RobotNavigateNLP
OSU 5522 AI project: navigate robots with natural language instructions

Duty | Who | Detail
------------ | ------------- | ------------
Understand Tensor flow | Shuang Li  | Mar 5 
Understand data set | Qi Zhu  | Mar 5

## matrix
- [x, y, d], d = 0, 90, 180, 270
- x, y表示一个点，d表示方向，0表示右
- [x, y, d] = object(6 * 自前后左右), floor + wall(11 * 前后左右) 

## function
- input: [x0, y0, d0], operation (0 = no action, 1 = go forward, 2 = turn right, 3 = turn left, 4 = turn back)
- output: [x1, y1, d1]
- if there is no way ahead, return null
