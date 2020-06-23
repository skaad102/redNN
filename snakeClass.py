import os
import pygame
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from DQN import DQNAgent
from random import randint
from keras.utils import to_categorical


#   parametros manualmente

def define_parameters():
    params = dict()
    params['epsilon_decay_linear'] = 1/70
    params['learning_rate'] = 0.0005
    params['capa1'] = 150   
    params['capa2'] = 150   
    params['capa3'] = 150    
    params['episodios'] = 150            
    params['memory_size'] = 2500
    params['batch_size'] = 500
    params['train'] = True
    params['load_trained_model'] = True
    params['save_dir'] = "trained_models/"
    params['load_dir']  = "trained_models/"
    params['frecuencia_guardado'] = 30
    return params


class Game:
    def __init__(self, game_width, game_height):
        pygame.display.set_caption('Juego SnakeNN')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.choque = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0

##Serpiente
class Player(object):
    def __init__(self, game):
        x = 0.5 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y
            

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]
        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # derecha - horiz
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # derecha - vert
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width - 40 \
                or self.y < 20 \
                or self.y > game.game_height - 40 \
                or [self.x, self.y] in self.position:
            game.choque = True
            #print("muerte choque :c")
        eat(self, food, game)
        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y
        if game.choque == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(200)

##Propiedades comida
class Food(object):
    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record

##Pantalla
def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)

def display_ui(game, score, record):
    fuente = pygame.font.SysFont('UI', 20)
    fuente_negrita = pygame.font.SysFont('UI', 20, True)
    text_score = fuente.render('PUNTAJE: ', True, (0, 0, 0))
    text_score_number = fuente.render(str(score), True, (0, 0, 0))
    text_highest = fuente.render('MEJOR PUNTAJE: ', True, (0, 0, 0))
    text_highest_number = fuente_negrita.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food, agent, batch_size):
    state_init1 = agent.get_state(game, player, food)  
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.choque)
    agent.remember(state_init1, action, reward1, state_init2, game.choque)
    agent.replay_new(agent.memory, batch_size)


def run(display_option, velocidad, params):
    pygame.init()
    agent = DQNAgent(params)
    contador_juegos = 0
    list_puntaje = []
    list_contador = []
    record = 0
    while contador_juegos < params['episodios']:
        #Cerrar Juego Con X Windows
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            ##
        
        game = Game(440, 440)
        s_player = game.player
        food1 = game.food

        # Realizar el primer movimiento
        initialize_game(s_player, game, food1, agent, params['batch_size'])
        if display_option:
            display(s_player, food1, game, record)

        while not game.choque:
            # agent.epsilon es configurado para dar aleatoriedad a las acciones
            agent.epsilon = 1 - (contador_juegos * params['epsilon_decay_linear'])

            # estado antigui
            state_old = agent.get_state(game, s_player, food1)

            # realizar acciones aleatorias basadas en agent.epsilon, o elegir la acción
            if randint(0, 1) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predecir acciones basadas en el estado anterior
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # realizar un nuevo movimiento y obtener un nuevo estado
            s_player.do_move(final_move, s_player.x, s_player.y, game, food1, agent)
            state_new = agent.get_state(game, s_player, food1)

            # establecer recompensa para el nuevo estado
            reward = agent.set_reward(s_player, game.choque)

            if params['train']:
                # entrenar memoria corta basada en la nueva acción y estado
                agent.train_short_memory(state_old, final_move, reward, state_new, game.choque)
                # almacenar los nuevos datos en una memoria a largo plazo
                agent.remember(state_old, final_move, reward, state_new, game.choque)

            record = get_record(game.score, record)
            if display_option:
                display(s_player, food1, game, record)
                pygame.time.wait(velocidad)
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        contador_juegos += 1
        print(f'Juego # {contador_juegos}      Puntaje: {game.score}')
        list_puntaje.append(game.score)
        list_contador.append(contador_juegos)
    #Graficar
    graficar_aprendizaje(list_contador, list_puntaje)


##Grafica
def graficar_aprendizaje(array_counter, array_score):
    sb.set(color_codes=True)
    ax = sb.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        color="b",
        x_jitter=.1,
        line_kws={'color': 'green'}
    )
    ax.set(xlabel='# Juegos', ylabel='Puntaje')
    plt.show()

if __name__ == '__main__':
    # Establecer opciones para activar o desactivar la vista del juego y su velocidad.
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", type=bool, default=True)
    parser.add_argument("--velocidad", type=int, default=20)
    args = parser.parse_args()
    run(args.display, args.velocidad, params)
