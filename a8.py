from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")
xor_training_data = [
    ([1, 1], [0]), 
    ([1, 0], [1]), 
    ([0, 1], [1]), 
    ([0, 0], [0])
    ]

xorn = NeuralNet(2, 2, 1)
xorn.train(xor_training_data)
print(xorn.test_with_expected(xor_training_data))

igt_training_data = [
    ([.9,.6,.8,.3,.1], [1.0]),
    ([.8,.8,.4,.6,.4], [1.0]),
    ([.7,.2,.3,.6,.3], [1.0]),
    ([.5,.5,.8,.4,.8], [0.0]),
    ([.3,.1,.6,.8,.8], [0.0]),
    ([.6,.3,.4,.3,.6], [0.0]),
    
]

igtn = NeuralNet(5, 3, 1)
igtn.train(igt_training_data)

print(igtn.evaluate([1.0,1.0,1.0,.1,.1]))
print(igtn.evaluate([.5,.2,.1,.7,.7]))
print(igtn.evaluate([.8,.3,.3,.3,.8]))
print(igtn.evaluate([.8,.3,.3,.8,.3]))
print(igtn.evaluate([.9,.8,.8,.3,.6]))



