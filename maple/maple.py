import numpy as np
from PIL import Image  # pillow
import sys

import pygame
import numpy as np
from .games.base.pygamewrapper import PyGameWrapper

class MAPLE(object):
    """ 
    TO DO: write description
    """

    def __init__(self,
                 game, fps=30, frame_skip=1, num_steps=1,
                 reward_values={}, force_fps=True, display_screen=False,
                 state_preprocessor=None, rng=24):

        self.game = game
        self.fps = fps
        self.frame_skip = frame_skip
        self.num_steps = num_steps
        self.force_fps = force_fps
        self.display_screen = display_screen

        self.last_action = []
        self.action = []
        self.previous_scores = np.zeros(self.game.n_agents)
        self.frame_count = 0

        # update the scores of games with values we pick
        if reward_values:
            self.game.adjustRewards(reward_values)


        self.rng = np.random.RandomState(rng)

        # some pygame games preload the images
        # to speed resetting and inits up.
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

        
        self.game.setRNG(self.rng)
        self.init()

        self.state_preprocessor = state_preprocessor
        self.state_dim = None

        if self.state_preprocessor is not None:
            self.state_dim = self.game.getGameState()

            if self.state_dim is None:
                raise ValueError(
                    "Asked to return non-visual state on game that does not support it!")
            else:
                self.state_dim = self.state_preprocessor(self.state_dim).shape

        if game.allowed_fps is not None and self.fps != game.allowed_fps:
            raise ValueError("Game requires %dfps, was given %d." %
                             (game.allowed_fps, self.fps))

    def _tick(self):
        """
        Calculates the elapsed time between frames or ticks.
        """
        if self.force_fps:
            return 1000.0 / self.fps
        else:
            return self.game.tick(self.fps)

    def init(self):
        """
        Initializes the game. This depends on the game and could include
        doing things such as setting up the display, clock etc.

        This method should be explicitly called.
        """
        self.game._setup()
        self.game.init() #this is the games setup/init

    def getActionsSet(self):
        """
        Gets the actions the game supports.

        Returns
        --------

        list of list of str
            for example: [["P1_NOOP", "P1_UP", "P1_DOWN"], ["P2_NOOP", "P2_UP", "P2_DOWN"]]

        """
        actions_set = self.game.actions_set

        return actions_set

    def getFrameNumber(self):
        """
        Gets the current number of frames the agent has seen
        since MAPLE was initialized.

        Returns
        --------

        int

        """

        return self.frame_count

    def game_over(self):
        """
        Returns True if the game has reached a terminal state and
        False otherwise.

        This state is game dependent.

        Returns
        -------

        bool

        """

        return self.game.game_over()

    def scores(self):
        """
        Gets the scores the agents currently have in game.

        Returns
        -------

        list of int

        """

        return self.game.getScores()

    def lives(self):
        """
        Gets the number of lives the agents have left. Not all games have
        the concept of lives.

        Returns
        -------

        list of int

        """

        return self.game.lives

    def reset_game(self):
        """
        Performs a reset of the games to a clean initial state.
        """
        self.last_actions = []
        self.actions = []
        self.previous_scores = np.zeros(self.game.n_agents)
        self.game.reset()

    def getScreenRGB(self):
        """
        Gets the current game screen in RGB format.

        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).


        """

        return self.game.getScreenRGB()

    def getScreenGrayscale(self):
        """
        Gets the current game screen in Grayscale format. Converts from RGB using relative lumiance.

        Returns
        --------
        numpy uint8 array
                Returns a numpy array with the shape (width, height).


        """
        frame = self.getScreenRGB()
        frame = 0.21 * frame[:, :, 0] + 0.72 * \
            frame[:, :, 1] + 0.07 * frame[:, :, 2]
        frame = np.round(frame).astype(np.uint8)

        return frame

    def saveScreen(self, filename):
        """
        Saves the current screen to png file.

        Parameters
        ----------

        filename : string
            The path with filename to where we want the image saved.

        """
        frame = Image.fromarray(self.getScreenRGB())
        frame.save(filename)

    def getScreenDims(self):
        """
        Gets the games screen dimensions.

        Returns
        -------

        tuple of int
            Returns a tuple of the following format (screen_width, screen_height).
        """
        return self.game.getScreenDims()

    def getGameStateDims(self):
        """
        Gets the games non-visual state dimensions.

        Returns
        -------

        tuple of int or None
            Returns a tuple of the state vectors shape or None if the game does not support it.
        """
        return self.state_dim

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        This can include items such as player position, velocity, ball location and velocity etc.

        Returns
        -------

        dict or None
            It returns a dict of game information. This greatly depends on the game in question and must be referenced against each game.
            If no state is available or supported None will be returned back.

        """
        state = self.game.getGameState()
        if state is not None and self.state_preprocessor is not None:
            return self.state_preprocessor(state)
        else:
            raise ValueError(
                "Was asked to return state vector for game that does not support it!")

    def act(self, actions):
        """
        Perform actions (1 by agent) on the game. We lockstep frames with actions. If act is not called the game will not run.

        Parameters
        ----------

        action : list(int)
            The indices of the actions we wish to perform. The index usually corresponds to the index item returned by getActionsSet().

        Returns
        -------

        int
            Returns the reward that the agent has accumlated while performing the action.

        """
        return sum(self._oneStepAct(actions) for i in range(self.frame_skip))
    
    def close(self):
        """
        Close the environment and display
        """
        pygame.quit()
#        sys.exit()

    def _draw_frame(self):
        """
        Decides if the screen will be drawn too
        """

        self.game._draw_frame(self.display_screen)

    def _oneStepAct(self, actions):
        """
        Performs actions on the game.
        """
        if self.game_over():
            return 0.0

        self._setActions(actions)
        for i in range(self.num_steps):
            time_elapsed = self._tick()
            self.game.step(time_elapsed)
            self._draw_frame()

        self.frame_count += self.num_steps

        return self._getRewards()

    def _setActions(self, actions):
        """
        Instructs the game to perform actions
        """
        self.game._setActions(actions)

    def _getRewards(self):
        """
        Returns the rewards the agents have gained as the difference between the last action and the current one.
        """
        scores = np.array(self.game.getScores())
        rewards = scores - self.previous_scores
        self.previous_scores = scores

        return rewards
