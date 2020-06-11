""" Red Neuronal para aprender a jugar snake """
""" Librerias """
import gym  
import gym_snake  
import numpy as np
import keyboard

gamma =  0.98
learning_rate = 5e-3
max_pasos = 1e6
global_pasos = 0
corriendo = True


env = gym.make('snake-v0')
env.grid_size = [15,15] ##Tamaño cuadricula
env.unit_size = 10
env.unit_gap = 1
env.snake_size = 3
env.n_foods = 1
observation = env.reset()
env.n_foods = 1

# Controlador
game_controller = env.controller

# Dibujo
grid_object = game_controller.grid
#
grid_pixels = grid_object.grid
    
# Snak
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]
arriba = snake_object1.UP
abajo = snake_object1.DOWN
izquierda = snake_object1.LEFT
derecha = snake_object1.RIGHT
while corriendo:

    accion = env.action_space.sample()
    if keyboard.is_pressed("LEFT"):
        env.step(izquierda)
    if keyboard.is_pressed("UP"):
        env.step(arriba)
    if keyboard.is_pressed("DOWN"):
        env.step(abajo)
    if keyboard.is_pressed("RIGHT"):
        env.step(derecha)
    env.render() 
    


env.close()

# """ entorno """
# env = gym.make('snake-v0')  
# """ numero de seldas """
# env.grid_size = [25,25] 
# """ numero de seldas """
# env.n_foods = 2 
# """ poner todo a 0 """
# observation = env.reset()  


# while True:
#     """ acciones aleatrias """
#     acccion = env.action_space.sample() 
#     env.step(acccion) 
#     """ pintar por pantalla """
#     env.render() 
