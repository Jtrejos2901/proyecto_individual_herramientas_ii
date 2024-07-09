import numpy as np
import random as rn
import pandas as pd
from deap import base as bs
from deap import creator as ct
from deap import tools as tl
from deap import algorithms as al
import matplotlib.pyplot as plt
from matplotlib import cm 

# Se crea la clase paraejecutar un algoritmo genético básico.
class GeneticAlgorithm:
    # Se crea el constructor de la clase.
    def __init__(self):
        '''
        Método contructor de la clase GeneticAlgorithm.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        # Se establecen los atributos propios.
        self.__toolbox = bs.Toolbox()
        self.__best_result = None
        self.__value = None
        self.__statistics = None
         
    # Método get del mejor resultado. 
    @property 
    def best_result(self):
        return self.__best_result
     
    # Método set del mejor resultado.
    @best_result.setter
    def best_result(self, new_val):
        self.__best_result = new_val  
     
    # Método get de las estadísticas. 
    @property 
    def statistics(self):
        return self.__statistics
     
    # Método set de las estadísticas. 
    @statistics.setter
    def statistics(self, new_results):
        self.__statistics = new_results  
    
    # Método get del valor del mejor resultado. 
    @property 
    def value(self):
        return self.__value
     
    # Método set del mejor resultado.
    @value.setter
    def value(self, new_val):
        self.__value = new_val  
         
    # Método que contiene la función objetivo.
    def target_function(self, individual):
        '''
        Método para que contien la función objetivo y realiza la evaluación de
        un individuo.

        Parameters:
            individual (list): Lista que contiene el genoma del individuo.
            
        Returns:
            result (tuple): El valor que posee el individuo evaluado en la 
            función objetivo. Si la evaluación se sale de los parámetros
            limites, se devuelve -1.
        '''
        for i in range(len(individual)):
            if individual[i] > 5.12 or individual[i] < -5.12:
                return 1000,
        
        result = (individual[0]**2 - 10 * np.cos(2 * np.pi * individual[0])) +\
                 (individual[1]**2 - 10 * np.cos(2 * np.pi * individual[1])) +\
                 20 
                 
        
        return result,
     
    # Método para la creación de clases de utilidad.
    def create(self):
        '''
        Método para la creación de las clases importantes en el algoritmo.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        ct.create("FitnessMax", bs.Fitness, weights = (-1.0,))
        ct.create("Individual", list, fitness = ct.FitnessMax)
       
    # Método para registrar las funciones de utilidad.
    def register(self, l_min, l_max, n_genom, n_pop):
        '''
        Método para el registro de funciones importantes para el algoritmo.

        Parameters:
            l_min (int): Valor mínimo para la generación de los genes 
                         aleatorios de los individuos.
            l_max (int): Valor máximo para la generación de los genes 
                         aleatorios de los individuos.
            n_genom (int): El número de genes que tendrán los individuos.
            n_pop (int): El número total de individuos de una población.
            
        Returns:
            None.
        '''
        # Se registra la función para la creación de genes.
        self.__toolbox.register("attr_uniform", rn.uniform, l_min, l_max)

        # Se registra la función que nos creará los genes y los guardará en un 
        # objeto de tipo Individual.
        self.__toolbox.register("individual", tl.initRepeat, ct.Individual, 
                                self.__toolbox.attr_uniform, n_genom)

        #Se registra la función que crea a la población.
        self.__toolbox.register("population", tl.initRepeat, list, 
                                self.__toolbox.individual, n_pop)
         
        # se registra la función fitness o target.
        self.__toolbox.register("evaluate", self.target_function)

        # Se registra la operación de cruce o mate.
        self.__toolbox.register("mate", tl.cxOnePoint)

        # Se registra la operación de mutación.
        self.__toolbox.register("mutate", tl.mutGaussian, mu=0, sigma=5, 
                                indpb=0.1)

        # Se registra la operación de selección.
        self.__toolbox.register("select", tl.selTournament, tournsize=3)
        
    # Método para generar un solo individuo.
    def generate_individual(self):
        '''
        Método para generar un individuos con los parámetros introducidos en el
        método "register".

        Parameters:
            None.
            
        Returns:
            None.
        '''
        return self.__toolbox.individual()
    
    # Método para generar una población.
    def generate_population(self):
        '''
        Método para generar una población con los parámetros introducidos en el
        método "register".

        Parameters:
            None.
            
        Returns:
            None.
        '''
        return pd.DataFrame(self.__toolbox.population())
        
    # Método para la ejecución del algoritmo.
    def execute(self, seed, CXPB, MUTPB, NGEN):
        '''
        Método para ejecución del algoritmo genético.
        
        Parameters:
            seed (int): Valor de la semilla para la reproducibilidad de 
                        resultados.
            CXPB (float): Probabilidad de cruce.
            MUTPB (float): Probabilidad de mutación.
            NGEN (int): Número de generaciones a generar.
            
        Returns:
            self.best_result (float): Mejor resultado.
            self.statistics(pandas.DataFrame): Estadísticas.
        '''
        # Se define la semilla para la reproducibilidad de los resultados.
        rn.seed(seed)
        
        # Se crea la población.
        pop = self.__toolbox.population()
        
        # Se inicializa el muro de la fama que contendrá al mejor individuo.
        hof = tl.HallOfFame(1)
        
        # Se inicializa el objeto que contendrá las estadísticas de las 
        # las iteraciones del algoritmo.
        stats = tl.Statistics(lambda ind: ind.fitness.values)
        
        # Se registran las funciones de interés para el objeto "stats".
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Se inicializa el objeto logbook que conedrá los resultados pora cada
        # iteración realizada.
        logbook = tl.Logbook()
        
        # Se ejecuta el algoritmo
        pop, logbook = al.eaSimple(pop, self.__toolbox, 
                                   cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                                   stats=stats, halloffame=hof, 
                                   verbose=False)
        
        # Se guardan los resultados en los atributos de clase.
        self.__best_result = pd.DataFrame({"x": [hof[0][0]], "y": [hof[0][1]]})
        self.__value = pd.DataFrame({"Mínimo aproximado": [hof[0].fitness.values[0]]})
        df = pd.DataFrame(logbook)
        df = df.rename({'0': 'x', '1': 'y'}, axis=1)
        self.__statistics = df
        
    # Método para gráficar la función objetivo.
    def plot_target_function(self):
        '''
        Método para graficar la función objetivo.
        
        Parameters:
            None.
            
        Returns:
            None.
        '''
        # Genra los limites.
        X = np.linspace(-5.12, 5.12, 100)     
        Y = np.linspace(-5.12, 5.12, 100)     
        X, Y = np.meshgrid(X, Y) 
        
        # Se obtiene el eje Z.
        Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
          (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20

        # Se crea la figura en 3D.
        fig = plt.figure(figsize=(16,6), dpi=350)  # Tamaño de la figura y resolución
        ax = fig.add_subplot(111, projection='3d') 

        # Se grafica la superficie.
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
          cmap=cm.nipy_spectral, linewidth=0.08,
          antialiased=True)    

        # Se añaden etiquetas al gráfico.
        ax.set_title('Gráfico de Superficie de la Función Rastrigin', fontsize=20)
        ax.set_xlabel('Eje X', fontsize=15)
        ax.set_ylabel('Eje Y', fontsize=15)
        ax.set_zlabel('Eje Z', fontsize=15)
        
        # Se muestra el gráfico.
        plt.show()

    # Método str de la clase.
    def __str__(self):
        '''
        Método para la impresión.

        Parameters:
            None.
            
        Returns:
            None.
        '''
        return f'El óptimo aproximado es: {self.best_result}.'
        
