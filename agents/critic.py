from keras import layers, models, optimizers, regularizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self,
                 state_size,
                 action_size,
                 net_cells,
                 learning_rate = 0.001):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model(learning_rate, net_cells)

    def build_model(self, learning_rate, net_cells_list):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape = (self.state_size,), name = 'states')
        actions = layers.Input(shape = (self.action_size,), name = 'actions')
        net_states = states
        net_actions = actions

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units = net_cells_list[0],
                                  activation = 'relu',
                                  kernel_regularizer = regularizers.l2(0.01)
                                  )(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        ii = 1
        net_states = layers.Dense(units = net_cells_list[ii],
                                  activation = 'relu',
                                  kernel_regularizer = regularizers.l2(0.01),
                                  name = 'net_states' + str(ii)
                                  )(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units = net_cells_list[ii],
                                   activation = 'relu',
                                   kernel_regularizer = regularizers.l2(0.01),
                                  name = 'net_actions' + str(ii)
                                   )(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Dense(units = net_cells_list[-1],
                           activation='relu',
                           kernel_regularizer=regularizers.l2(0.01)
                           )(net)
        net = layers.Activation('relu')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.5)(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units = 1,
                                name='q_values',
                                kernel_regularizer = regularizers.l2(0.01)
                                )(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr = learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        
