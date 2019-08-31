
def initialize():
    settings = {  
        "activation" : "softmax",
        "embedding_size" : 100
    } 
    return settings

def fit():
    settings = {  
        "epochs" : 150,
        "verbose" : False,   
        "batch_size" : 128       
    } 
    return settings

def compile():
    settings = {
        "loss" : "categorical_crossentropy",
        "optimizer" : "rmsprop",
        "metrics" : ["accuracy"]
    }
    return settings 
