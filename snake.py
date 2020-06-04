""" Red Neuronal para aprender a jugar snake """
""" Librerias """
import gym  
import gym_snake 

""" entorno """
env = gym.make('snake-v0')  
""" numero de seldas """
env.grid_size = [25,25] 
""" numero de seldas """
env.n_foods = 2 
""" poner todo a 0 """
observation = env.reset()  


while True:
    """ acciones aleatrias """
    acccion = env.action_space.sample() 
    env.step(acccion) 
    """ pintar por pantalla """
    env.render() 
