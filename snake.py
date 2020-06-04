""" Red Neuronal para aprender a jugar snake """
import gym """ libreria de juego """
import gym_snake""" libreria de snake """


env = gym.make('snake-v0')  """ entorno """
env.grid_size = [25,25] """ numero de seldas """
env.n_foods = 2 """ numero de comidas """
observation = env.reset()  """ poner todo a 0 """


while True:
    acccion = env.action_space.sample() 
    env.step(acccion) """ acciones aleatrias """
    env.render() """ pintar por pantalla """
