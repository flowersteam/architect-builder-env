import numpy as np
import sys

sys.path.append('..')
from gym import Env, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from env_comem.utils import render_image, permutation

## AGENT ACTIONS

# 0 : DO_NOTHING
# 1 : LEFT
# 2 : RIGHT
# 3 : UP
# 4 : DOWN
# 5 : TOGGLE (TAKE or DROP)

DO_NOTHING = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
TOGGLE = 5

ACTIONS_DIRECTION_DICT = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
PRINT_ACTION_DICT = {0: 'x', 1: '\u2190', 2: '\u2192', 3: '\u2191', 4: '\u2193', 5: 'I/O'}

LIST_OF_GOAL = ['horizontal_line', 'vertical_line', 'grasp_object', 'place_object']

DICT_OF_SHAPE_POSITIONS = {'u': [(1, 3), (1, 2), (2, 2), (3, 2), (4, 2), (4, 3)],
                           'c': [(3, 1), (2, 1), (2, 2), (2, 3), (2, 4), (3, 4)],
                           'l': [(4, 1), (3, 1), (2, 1), (2, 2), (2, 3), (2, 4)],
                           't': [(2, 1), (2, 2), (2, 3), (1, 3), (3, 3), (2, 4)],
                           'T': [(2, 1), (2, 2), (2, 3), (2, 4), (1, 4), (3, 4)],
                           'y': [(0, 4), (1, 3), (2, 2), (2, 1), (3, 3), (4, 4)],
                           'o': [(3, 1), (2, 2), (2, 3), (3, 4), (4, 3), (4, 2)],
                           'diag': [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)],
                           }

LIST_OF_SHAPES = list(DICT_OF_SHAPE_POSITIONS.keys())


# todo: note that for 'grasp_object' the agent can grasp any object it wants
# todo: and that for 'place_object' it can also place any object it wants
# todo: this might make these tasks not general enough to be composed into building lines


class Object(object):
    def __init__(self, type, coordinate):
        self.type = type
        self.coordinate = np.asarray(coordinate)
        self.is_grasped = False


class Grasper(object):
    def __init__(self, coordinate):
        self.coordinate = np.asarray(coordinate)
        self.grasped_object = None

    def grasp(self, object):
        assert isinstance(object, Object)
        assert not object.is_grasped
        assert self.grasped_object is None
        assert all(self.coordinate == object.coordinate)

        self.grasped_object = object
        object.is_grasped = True

    def drop(self):
        assert self.grasped_object is not None
        assert self.grasped_object.is_grasped

        self.grasped_object.is_grasped = False
        self.grasped_object = None

    @property
    def is_grasping(self):
        return self.grasped_object is not None


class Buildworld(Env):
    def __init__(self, n_objects, grid_size, change_goal, obs_type, verbose, seed, goal=None, save_fig=False):
        self.n_objects = n_objects
        if not isinstance(grid_size, (list, tuple)) or not len(grid_size) == 2:
            raise ValueError("grid_size argument must be a list/tuple of length 2")

        self.change_goal = change_goal

        self._action_direction_dict = ACTIONS_DIRECTION_DICT
        # we revert the action to direction dict
        self._direction_action_dict = {value: key for key, value in ACTIONS_DIRECTION_DICT.items()}

        self.action_penalty = 0.

        self.action_space = spaces.Discrete(len(self._action_direction_dict) + 1)

        self._grid_size = tuple(grid_size)
        # x_lim and y_lim are one above maximum possible values such that x < x_lim and y < y_lim
        self._x_lim, self._y_lim = self._grid_size

        self.obs_type = obs_type
        self.compute_observation_space()

        self.verbose = verbose
        self.save_fig = save_fig

        self.this_fig_num = 0
        if self.verbose == True:
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')
            self.i = 0
        self.seed(seed)

        self.objects = None
        self.grasper = None
        self.goal = goal

    @property
    def entities(self):
        return [self.grasper] + self.objects

    def initialize_grasper(self):
        return Grasper(coordinate=self.sample_random_coordinate())

    def initialize_objects(self):

        objects = []
        objects_coordinates = []

        for i in range(self.n_objects):
            init_coordinates = self.sample_random_coordinate()

            # we cannot have superposing objects
            same_coordinates = [all(init_coordinates == coord) for coord in objects_coordinates]
            while any(same_coordinates):
                init_coordinates = self.sample_random_coordinate()
                same_coordinates = [all(init_coordinates == coord) for coord in objects_coordinates]

            objects_coordinates.append(init_coordinates)
            objects.append(Object(type=1, coordinate=init_coordinates))

        return objects

    def get_init_state(self):
        grasper = self.initialize_grasper()
        objects = self.initialize_objects()
        entities = [grasper] + objects
        return self.encode_obs(entities, to_obs_type=self.obs_type)

    def get_random_state(self):
        # we get a random state by running a random policy from an initial state
        grasper = self.initialize_grasper()
        objects = self.initialize_objects()
        entities = [grasper] + objects

        len_of_random_action_seq = 5
        random_action_list = self.np_random.randint(low=0, high=6, size=len_of_random_action_seq)
        for action in random_action_list:
            self._engine_transition_fct(entities, action)

        return self.encode_obs(entities, to_obs_type=self.obs_type)

    def create_measurement_states(self, n_states):
        return np.asarray([self.get_random_state() for _ in range(n_states)])

    def compute_observation_space(self):
        if self.obs_type == 'xy_continuous':
            # (x/(x_lim -1), y/(y_lim-1)) continuous coordinates for grasper and each object
            # 0. if not grasped or 1. if grasped for each object and 0./1. for grasper.is_gasping
            high = np.array([1.] * (3 * self.n_objects + 3))
            low = 0. * high
            self.observation_space = spaces.Box(low=low, high=high)

        elif self.obs_type == 'tile':
            # matrix of size (x_lim, y_lim, 3)
            # (x,y,0) indicates type of block on the ground (0 if no object)
            # (x,y,1) indicates bool(grasper is here)
            # (x,y,2) indicates type of block in the grasper ((x,y,2) != 0 <=> (x,y,1) = 1)
            high = np.ones((self._x_lim, self._y_lim, 3))
            self.observation_space = spaces.Box(low=0. * high, high=high)

        elif self.obs_type == 'tile_1D':
            # same as tile but flatten
            high = np.ones((self._x_lim * self._y_lim * 3))
            self.observation_space = spaces.Box(low=0. * high, high=high)

        else:
            raise NotImplementedError

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def sample_random_coordinate(self):
        return self.np_random.randint((0, 0), self._grid_size)

    def sample_random_goal_type(self):
        return self.np_random.choice(LIST_OF_GOAL)

    def _engine_reset(self):
        self.objects = self.initialize_objects()
        self.grasper = self.initialize_grasper()
        if self.goal is None or self.change_goal:
            self.goal = self.np_random.choice(LIST_OF_GOAL)

        if 'place_object' in self.goal:
            self.object_destination = self.sample_random_coordinate()
            self.goal = f'place_object_({self.object_destination})'

        if 'make_shape' in self.goal:
            self.shape_id = self.np_random.randint(0, len(LIST_OF_SHAPES))
            self.goal = f'make_shape_{LIST_OF_SHAPES[self.shape_id]}'

    def reset(self):
        self._engine_reset()
        return self.encode_obs(self.entities, to_obs_type=self.obs_type), self.goal

    def _clamp_to_grid(self, coordinate):
        return np.asarray([max(0, min(coordinate[0], self._x_lim - 1)), max(0, min(coordinate[1], self._y_lim - 1))])

    def detect_straight_object_line(self, objects, orientation):
        coordinates = [obj.coordinate for obj in objects]
        x_list, y_list = [[coord[i] for coord in coordinates] for i in range(2)]

        if orientation == 'horizontal':
            invariant_list = y_list
            growing_list = x_list
        elif orientation == 'vertical':
            invariant_list = x_list
            growing_list = y_list
        else:
            raise NotImplementedError

        if len(set(invariant_list)) == 1:
            # all objects have the same invariant-coordinate
            growing_list = sorted(growing_list)
            if growing_list[0] == growing_list[1] - 1 == growing_list[2] - 2:
                return True

        return False

    def detect_object_at_destination(self, objects, destination):
        is_at_dest = [all(obj.coordinate == destination) for obj in objects if not obj.is_grasped]
        return any(is_at_dest)

    def _engine_reward_fct(self, state, action):
        grasper = state[0]
        objects = state[1:]

        reward = 0.

        if 'line' in self.goal:
            # all objects must be on the ground
            if not grasper.is_grasping:
                if self.goal == 'horizontal_line':
                    reward += float(self.detect_straight_object_line(objects, 'horizontal'))
                elif self.goal == 'vertical_line':
                    reward += float(self.detect_straight_object_line(objects, 'vertical'))
                else:
                    raise ValueError
        elif self.goal == 'grasp_object':
            reward += float(grasper.is_grasping)

        elif 'place_object' in self.goal:
            reward += float(self.detect_object_at_destination(objects, self.object_destination))
        elif 'make_shape' in self.goal:

            reward = float(self.on_shape(grasper, objects, DICT_OF_SHAPE_POSITIONS[LIST_OF_SHAPES[self.shape_id]]))
        else:
            raise NotImplementedError

        ## Add action penalty
        if not action == DO_NOTHING:
            reward -= self.action_penalty

        return reward

    def get_correctly_placed(self, grasper, objects, shape_positions):
        object_positions = [obj.coordinate for obj in objects]

        for i, obj in enumerate(objects):
            obj._id = i

        # compute the distance for all object, position pair
        distances = np.zeros((len(object_positions), len(shape_positions)))
        correclty_placed = 0
        for i, obj_coord in enumerate(object_positions):
            for j, dest_coord in enumerate(shape_positions):
                distances[i, j] = self.compute_distance_coordinate(obj_coord, dest_coord)
                if distances[i, j] == 0.:
                    correclty_placed += 1.
                    if not objects[i].is_grasped:
                        correclty_placed += 1.

        return - correclty_placed

    def get_correctly_placed(self, grasper, objects, shape_positions):
        object_positions = [obj.coordinate for obj in objects]

        for i, obj in enumerate(objects):
            obj._id = i

        # compute the distance for all object, position pair
        distances = np.zeros((len(object_positions), len(shape_positions)))
        correclty_placed = 0
        for i, obj_coord in enumerate(object_positions):
            for j, dest_coord in enumerate(shape_positions):
                distances[i, j] = self.compute_distance_coordinate(obj_coord, dest_coord)
                if distances[i, j] == 0.:
                    correclty_placed += 1.
                    if not objects[i].is_grasped:
                        correclty_placed += 1.

        return correclty_placed

    def on_shape(self, grasper, objects, shape_positions):
        object_positions = [obj.coordinate for obj in objects]

        for i, obj in enumerate(objects):
            obj._id = i

        # compute the distance for all object, position pair
        distances = np.zeros((len(object_positions), len(shape_positions)))
        on_it = []
        for i, obj_coord in enumerate(object_positions):
            current_on = False
            for j, dest_coord in enumerate(shape_positions):
                distances[i, j] = self.compute_distance_coordinate(obj_coord, dest_coord)
                if distances[i, j] == 0.:
                    current_on = True
            on_it.append(current_on)
        return all(on_it) and not grasper.is_grasping

    def reward_fct(self, obs, action, next_obs):
        # only looks at next state
        next_state = self.decode_obs(next_obs, from_obs_type=self.obs_type)
        return self._engine_reward_fct(next_state, action)

    def move(self, grasper, action):
        coordinates = grasper.coordinate
        next_coordinates = self._clamp_to_grid([coordinates[0] + self._action_direction_dict[action][0],
                                                coordinates[1] + self._action_direction_dict[action][1]])
        grasper.coordinate = next_coordinates
        if grasper.is_grasping:
            grasper.grasped_object.coordinate = next_coordinates

    def toggle(self, grasper, objects):
        # the number of objects at the same location than the grasper will influence what is done
        n_at_location = 0
        at_location = []
        for obj in objects:
            if all(obj.coordinate == grasper.coordinate):
                n_at_location += 1
                at_location.append(obj)

        if grasper.is_grasping:
            # we will try to drop the grasped object if there is space in the current coordinates
            # if there is only one object at the grasper's coordinate it means that it can drop the object
            # (only object at that location is the grasped one)
            assert n_at_location >= 1

            if n_at_location == 1:
                grasper.drop()
        else:
            # if there is an object below the grasper we will grasp it
            assert n_at_location <= 1

            if n_at_location == 1:
                grasper.grasp(at_location[0])

    def heuristic_policy(self, obs):
        state = self.decode_obs(obs, from_obs_type=self.obs_type)
        action = self._engine_heuristic_policy(state, self.goal)
        return action

    def heuristic_value(self, obs, discount, return_steps=False, use_approx=True):
        state = self.decode_obs(obs, from_obs_type=self.obs_type)
        return self._engine_heuristic_value(state, self.goal, discount, return_steps, use_approx)

    def _engine_heuristic_policy(self, state, goal):
        grasper = state[0]
        objects = state[1:]

        if goal == 'grasp_object':
            return self.grasp_closest_object(grasper, objects)
        elif 'line' in goal:
            if 'vertical' in goal:
                return self.build_line(grasper, objects, 'vertical')
            elif 'horizontal' in goal:
                return self.build_line(grasper, objects, 'horizontal')
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def get_base_object(self, objects, dl):
        # finds the base of the line (bloc that is closer to the other blocks and that will not move)
        potential_base_object = [obj for obj in objects if not obj.is_grasped]
        distances_to_other_objects = [[self.compute_distance(potential_base, obj) for obj in objects]
                                      for potential_base in potential_base_object]
        summed_distances_to_other_objects = [sum(distances) for distances in distances_to_other_objects]

        min_dist = np.min(summed_distances_to_other_objects)
        min_indexes = np.where(summed_distances_to_other_objects == min_dist)[0]

        if len(min_indexes) == 1:
            base_idx = min_indexes[0]
        else:
            closest_objs = [potential_base_object[i] for i in min_indexes]
            closest_obj_alignements = [[np.abs(np.dot(closest.coordinate - obj.coordinate, dl)) for obj in objects]
                                       for closest in closest_objs]
            closest_obj_alignements_sums = [np.sum(alig) for alig in closest_obj_alignements]
            # we use argmax because we do not want them aligned in the direction of the line
            max_idx = np.argmax(closest_obj_alignements_sums)
            base_idx = min_indexes[max_idx]

        base_object = potential_base_object.pop(base_idx)
        return base_object, potential_base_object

    def build_line(self, grasper, objects, orientation):

        if orientation == 'vertical':
            dl = np.array([0., 1.])
        elif orientation == 'horizontal':
            dl = np.array([1., 0.])
        else:
            raise NotImplementedError

        base_object, other_objects = self.get_base_object(objects, dl)

        # two situation now
        if grasper.is_grasping:
            # todo: check some failure case where the agent does not drop the object
            # we have to come and place the object next to the base
            side_of_the_base = np.sign(np.dot((grasper.coordinate - base_object.coordinate), dl))
            was_aligned = False
            if side_of_the_base == 0:
                was_aligned = True
                # take the side where there is more room and is available
                if np.dot(np.array([self._x_lim, self._y_lim]) / 2, dl) > np.dot(base_object.coordinate, dl):
                    side_of_the_base = 1.
                else:
                    side_of_the_base = -1.

            potential_coordinate = base_object.coordinate + side_of_the_base * dl
            # we try the closest side
            if self.is_available(potential_coordinate, objects):
                return self.drop_at(grasper, potential_coordinate)
            else:
                # if we were aligned the other side is also closest
                if was_aligned:
                    potential_coordinate = base_object.coordinate - side_of_the_base * dl
                    if self.is_available(potential_coordinate, objects):
                        return self.drop_at(grasper, potential_coordinate)

                potential_coordinate = base_object.coordinate + 2 * side_of_the_base * dl
                assert self.is_available(potential_coordinate, objects)
                return self.drop_at(grasper, potential_coordinate)

        else:

            if self.detect_straight_object_line(objects, orientation=orientation):
                return DO_NOTHING

            # we have to go and pick the next object we want to place
            # we go for the closest object that is not placed
            grasper_dist_to_obj = [self.compute_distance(grasper, obj) for obj in other_objects]
            closest_idx = np.argmin(grasper_dist_to_obj)
            closest_obj = other_objects.pop(closest_idx)
            # the object we want to grab is already well placed
            if all(closest_obj.coordinate + dl == base_object.coordinate) or all(
                    closest_obj.coordinate - dl == base_object.coordinate):
                to_grab = other_objects.pop()
            else:
                to_grab = closest_obj

            return self.go_grasp(grasper, to_grab)

    def go_grasp(self, grasper, object):
        direction = self.get_direction_from_to(grasper, object)
        if all(direction == np.array([0., 0.])):
            return TOGGLE
        else:
            return self._direction_action_dict[tuple(direction)]

    def steps_to_go_grasp_and_update(self, grasper, object):
        distance = self.compute_distance(grasper, object)
        grasper.coordinate = object.coordinate
        grasper.grasp(object)
        return distance + 1

    def drop_at(self, grasper, coordinate):
        direction = self.get_direction_from_to(grasper, coordinate, to_is_coordinate=True)
        if all(direction == np.array([0., 0.])):
            return TOGGLE
        else:
            return self._direction_action_dict[tuple(direction)]

    def steps_to_drop_at_with_update(self, grasper, coordinate):
        distance = self.compute_distance_coordinate(grasper.coordinate, coordinate)
        grasper.coordinate = coordinate
        grasper.grasped_object.coordinate = coordinate
        grasper.drop()

        return distance + 1

    def is_available(self, coordinate, objects):
        if (coordinate[0] < 0 or self._x_lim <= coordinate[0]) or (
                coordinate[1] < 0 or self._y_lim <= coordinate[1]):
            return False
        else:
            for obj in objects:
                if not obj.is_grasped:
                    if all(obj.coordinate == coordinate):
                        return False
        return True

    def _engine_heuristic_value(self, state, goal, discount, return_steps=False, use_approx=True):

        grasper = state[0]
        objects = state[1:]

        if goal == 'grasp_object':
            steps_to_reach_goal = self.steps_to_grasp_closest_object(grasper, objects)

        elif 'line' in goal:
            if 'vertical' in goal:
                orientation = 'vertical'
            elif 'horizontal' in goal:
                orientation = 'horizontal'
            else:
                raise NotImplementedError

            if use_approx:
                steps_to_reach_goal = self.steps_to_build_line_approximated(grasper, objects, orientation)
            else:
                steps_to_reach_goal = self.steps_to_build_line(grasper, objects, orientation)

        elif 'place_object' in goal:
            destination = np.array([int(s) for s in goal if s.isdigit()])
            steps_to_reach_goal = self.steps_to_place_object(grasper, objects, destination)
        elif 'make_shape' in goal:
            steps_to_reach_goal = 0
            # todo: compute mappings and isolate object to call place
            shape_positions = DICT_OF_SHAPE_POSITIONS[LIST_OF_SHAPES[self.shape_id]]
            object_positions = [obj.coordinate for obj in objects]
            objects_ids = {}
            for i, obj in enumerate(objects):
                obj.id = i
                objects_ids[i] = obj

            dones = {obj.id: False for obj in objects}
            mappings, _ = self.get_object_shape_mapping(object_positions, shape_positions)

            if grasper.is_grasping:
                obj = grasper.grasped_object
                destination = shape_positions[mappings[obj.id]]
                steps_to_reach_goal += self.steps_to_place_object(grasper, [obj], destination)
                grasper.coordinate = destination
                grasper.drop()
                dones[obj.id] = True

            for obj in objects:
                if all(obj.coordinate == shape_positions[mappings[obj.id]]):
                    dones[obj.id] = True

            while not all(list(dones.values())):
                remaining_objects = [obj for obj in objects if not dones[obj.id]]
                remaining_objects_pos = [obj.coordinate for obj in remaining_objects]
                remaining_distances = [self.compute_distance_coordinate(grasper.coordinate, coord)
                                       for coord in remaining_objects_pos]

                if len(remaining_distances) == 1:
                    id_min = 0
                else:
                    id_min = np.argmin(remaining_distances)

                next_obj = remaining_objects[id_min]
                destination = shape_positions[mappings[next_obj.id]]
                steps_to_reach_goal += self.steps_to_place_object(grasper, [next_obj], destination)
                grasper.coordinate = destination
                dones[next_obj.id] = True
        else:
            raise NotImplementedError

        if steps_to_reach_goal > 0:
            before_reach = - self.action_penalty * (1 - discount ** (steps_to_reach_goal - 1)) / (1 - discount)
            to_reach = (1. - self.action_penalty) * discount ** (steps_to_reach_goal - 1)
        else:
            before_reach = 0.
            to_reach = 0.

        once_reached = discount ** steps_to_reach_goal / (1 - discount)

        value = before_reach + to_reach + once_reached

        if return_steps:
            return value, steps_to_reach_goal
        else:
            return value

    def get_object_shape_mapping(self, object_positions, shape_positions):

        # compute the distance for all object, position pair
        distances = np.zeros((len(object_positions), len(shape_positions)))
        for i, obj_coord in enumerate(object_positions):
            for j, dest_coord in enumerate(shape_positions):
                distances[i, j] = self.compute_distance_coordinate(obj_coord, dest_coord)

        # all possible object permutation
        object_permutations = permutation(list(range(len(object_positions))))

        # all possible mappings between object and position
        mappings = []
        for perm in object_permutations:
            maping = {}
            for i, j in zip(perm, list(range(len(shape_positions)))):
                maping[i] = j
            mappings.append(maping)

        # we compute the distance to cover of each mapping
        mappings_distance = []
        for maping in mappings:
            dist = 0
            for i, j in maping.items():
                dist += distances[i, j]
            mappings_distance.append(dist)
        min_map = np.argmin(mappings_distance)

        return mappings[min_map], distances

    def steps_to_place_object(self, grasper, objects, destination):
        if self.detect_object_at_destination(objects, destination):
            return 0
        else:
            if grasper.is_grasping:
                # if we are already grasping an object we just have to go and drop it
                steps = self.compute_distance_coordinate(grasper.coordinate, destination)
                # we add one for the drop action
                steps += 1
                return steps
            else:
                # we will pick the object that has the grasper cover less distance
                distances = [sum([self.compute_distance_coordinate(obj.coordinate, grasper.coordinate),
                                  self.compute_distance_coordinate(obj.coordinate, destination)]) for obj in objects]
                steps = min(distances)
                # we add pick and drop actions
                steps += 2
                return steps

    def steps_to_build_line(self, grasper, objects, orientation):
        steps = 0
        while grasper.is_grasping or (not self.detect_straight_object_line(objects, orientation)):
            action = self.build_line(grasper, objects, orientation)
            self._engine_transition_fct([grasper] + objects, action)
            steps += 1
        return steps

    def steps_to_build_line_approximated(self, grasper, objects, orientation):

        if orientation == 'vertical':
            dl = np.array([0., 1.])
        elif orientation == 'horizontal':
            dl = np.array([1., 0.])
        else:
            raise NotImplementedError

        steps = 0

        while grasper.is_grasping or (not self.detect_straight_object_line(objects, orientation)):

            base_object, other_objects = self.get_base_object(objects, dl)

            # two situation now
            if grasper.is_grasping:
                # we have to come and place the object next to the base
                side_of_the_base = np.sign(np.dot((grasper.coordinate - base_object.coordinate), dl))
                was_aligned = False
                if side_of_the_base == 0:
                    was_aligned = True
                    # take the side where there is more room and is available
                    if np.dot(np.array([self._x_lim, self._y_lim]) / 2, dl) > np.dot(base_object.coordinate, dl):
                        side_of_the_base = 1.
                    else:
                        side_of_the_base = -1.

                potential_coordinate = base_object.coordinate + side_of_the_base * dl
                # we try the closest side
                if self.is_available(potential_coordinate, objects):

                    steps += self.steps_to_drop_at_with_update(grasper, potential_coordinate)

                else:
                    # if we were aligned the other side is also closest
                    if was_aligned:
                        potential_coordinate = base_object.coordinate - side_of_the_base * dl
                        if self.is_available(potential_coordinate, objects):
                            steps += self.steps_to_drop_at_with_update(grasper, potential_coordinate)
                            continue

                    potential_coordinate = base_object.coordinate + 2 * side_of_the_base * dl
                    assert self.is_available(potential_coordinate, objects)
                    steps += self.steps_to_drop_at_with_update(grasper, potential_coordinate)

            else:

                # we have to go and pick the next object we want to place
                # we go for the closest object that is not placed
                grasper_dist_to_obj = [self.compute_distance(grasper, obj) for obj in other_objects]
                closest_idx = np.argmin(grasper_dist_to_obj)
                closest_obj = other_objects.pop(closest_idx)
                # the object we want to grab is already well placed
                if all(closest_obj.coordinate + dl == base_object.coordinate) or all(
                        closest_obj.coordinate - dl == base_object.coordinate):
                    to_grab = other_objects.pop()
                else:
                    to_grab = closest_obj

                steps += self.steps_to_go_grasp_and_update(grasper, to_grab)
        return steps

    def grasp_closest_object(self, grasper, objects):
        if grasper.is_grasping:
            # if an object is already grasp there is nothing to do
            return DO_NOTHING
        else:
            distance_to_objects = np.asarray([self.compute_distance(grasper, obj) for obj in objects])
            closest_idx = np.argmin(distance_to_objects)
            closest_object = objects[closest_idx]
            min_distance = distance_to_objects[closest_idx]
            if min_distance == 0.:
                # we are above an object, we just have to grasp it
                return TOGGLE
            else:
                direction = self.get_direction_from_to(grasper, closest_object)
                action = self._direction_action_dict[tuple(direction)]
                return action

    def steps_to_grasp_closest_object(self, grasper, objects):
        if grasper.is_grasping:
            # if an object is already grasp there is nothing to do
            return 0
        else:
            distance_to_objects = np.asarray([self.compute_distance(grasper, obj) for obj in objects])
            min_distance = np.amin(distance_to_objects)
            # we have to go there and then grasp it
            return min_distance + 1

    def get_direction_from_to(self, from_entity, to_entity, to_is_coordinate=False):
        if to_is_coordinate:
            to_coord = to_entity
        else:
            to_coord = to_entity.coordinate

        diff_vec = to_coord - from_entity.coordinate

        if all(diff_vec == np.array([0., 0.])):
            return diff_vec
        else:
            principal_dir = np.argmax(np.abs(diff_vec))
            direction = np.zeros_like(diff_vec)
            direction[principal_dir] = np.sign(diff_vec[principal_dir])
            return direction

    def compute_distance(self, entity1, entity2):
        return np.sum(np.abs(entity1.coordinate - entity2.coordinate))

    def compute_distance_coordinate(self, coord1, coord2):
        return np.sum(np.abs(coord1 - coord2))

    def _engine_transition_fct(self, state, action):
        # here the engine state is an entity list be careful that this function does not return the next
        # state but directly modifies the entities

        grasper = state[0]
        objects = state[1:]

        if action in self._action_direction_dict:
            self.move(grasper, action)

        elif action == 5:
            self.toggle(grasper, objects)

        else:
            raise NotImplementedError

    def transition_fct(self, obs, action):
        # note that here the state is directly modified (but not the obs since it is decoder and then
        # encoded)
        state = self.decode_obs(obs, from_obs_type=self.obs_type)
        self._engine_transition_fct(state, action)
        return self.encode_obs(state, to_obs_type=self.obs_type)

    def step(self, action):
        info = {}
        action = int(action)
        self._engine_transition_fct(state=self.entities, action=action)

        reward = self._engine_reward_fct(self.entities, action)
        done = False  # we are in the infinite setting (with interaction time-limit) there is no final/absorbing state
        # and therefore done is always False
        nxt_obs = self.encode_obs(self.entities, self.obs_type)

        return nxt_obs, reward, done, info

    def done(self, obs):
        return self.reward_fct(None, None, obs) > 0.

    def coordinates_to_ranges(self, coordinate):
        return np.asarray([coordinate[0] / (self._x_lim - 1), coordinate[1] / (self._y_lim - 1)])

    def ranges_to_coordinates(self, ranges):
        return np.asarray([ranges[0] * (self._x_lim - 1), ranges[1] * (self._y_lim - 1)])

    def encode_obs(self, state, to_obs_type):
        # todo: this encoding does not take into consideration object types yet
        # here the state is the entities list
        grasper = state[0]
        objects = state[1:]

        if to_obs_type == 'xy_continuous':

            grasper_xy = self.coordinates_to_ranges(grasper.coordinate)
            is_grasping = np.asarray([float(grasper.is_grasping)])

            objects_xy = [self.coordinates_to_ranges(obj.coordinate) for obj in objects]
            objects_xy = np.concatenate(objects_xy)
            objects_is_grasped = np.asarray([float(obj.is_grasped) for obj in objects])

            return np.concatenate((grasper_xy, is_grasping, objects_xy, objects_is_grasped))

        elif 'tile' in to_obs_type:
            # matrix of size (x_lim, y_lim, 3)
            # (x,y,0) indicates type of block on the ground (0 if no object)
            # (x,y,1) indicates bool(grasper is here)
            # (x,y,2) indicates type of block in the grasper ((x,y,2) != 0 <=> (x,y,1) = 1)

            tile_obs = np.zeros(shape=(self._grid_size[0], self._grid_size[1], 3))

            # grasper position
            tile_obs[grasper.coordinate[0], grasper.coordinate[1], 1] = 1.

            # grasped object is put to third tile
            if grasper.is_grasping:
                grasped_obj = grasper.grasped_object
                tile_obs[grasped_obj.coordinate[0], grasped_obj.coordinate[1], 2] = float(1)

            # floor object position
            for obj in objects:
                if not obj.is_grasped:
                    tile_obs[obj.coordinate[0], obj.coordinate[1], 0] = float(1)

            if '1D' in to_obs_type:
                return tile_obs.flatten()
            else:
                return tile_obs
        else:
            raise NotImplementedError

    def get_entities_from_list_descriptions(self, grasper_coordinates, is_grasping, objects_coordinates,
                                            objects_is_grasped):

        objects = [Object(type=1, coordinate=coord) for coord in objects_coordinates]
        grasper = Grasper(coordinate=grasper_coordinates)

        if is_grasping == 1.:
            grasped_idx = np.nonzero(objects_is_grasped == 1.)
            assert len(grasped_idx[0]) == 1
            grasper.grasp(objects[grasped_idx[0][0]])
        else:
            # no object is grasped
            assert not np.any(objects_is_grasped)
            # each object is on a different location
            assert len(set([tuple(coord) for coord in objects_coordinates])) == self.n_objects

        return [grasper] + objects

    def decode_obs(self, obs, from_obs_type):
        # todo: this encoding does not take into consideration object types yet
        if from_obs_type == 'xy_continuous':

            grasper_xy = obs[0:2]
            is_grasping = obs[2]

            objects_xy = obs[3:3 + 2 * self.n_objects]
            objects_is_grasped = obs[3 + 2 * self.n_objects:]
            assert len(objects_is_grasped) == self.n_objects

            grasper_coordinates = self.ranges_to_coordinates(grasper_xy)

            # objects_xy should be a list of x,y coordinates, one x,y pair per object
            objects_coordinates = [self.ranges_to_coordinates([objects_xy[2 * i], objects_xy[2 * i + 1]]) for i in
                                   range(len(objects_xy) // 2)]

            return self.get_entities_from_list_descriptions(grasper_coordinates, is_grasping,
                                                            objects_coordinates, objects_is_grasped)

        elif 'tile' in from_obs_type:
            if '1D' in from_obs_type:
                obs = obs.reshape((self._x_lim, self._y_lim, 3))

            grasper_coordinates = np.where(obs[:, :, 1] == 1.)
            grasper_coordinates = list(zip(grasper_coordinates[0], grasper_coordinates[1]))
            assert len(grasper_coordinates) == 1
            grasper_coordinates = grasper_coordinates[0]

            objects_coordinates = np.where(obs[:, :, 0] == 1.)
            objects_coordinates = list(zip(objects_coordinates[0], objects_coordinates[1]))
            objects_is_grasped = [0.] * len(objects_coordinates)

            grasped_object_coordinates = np.where(obs[:, :, 2] == 1.)
            grasped_object_coordinates = list(zip(grasped_object_coordinates[0], grasped_object_coordinates[1]))
            if len(grasped_object_coordinates) > 0:
                assert len(grasped_object_coordinates) == 1
                is_grasping = 1.
                objects_coordinates += grasped_object_coordinates
                objects_is_grasped += [1.]
            else:
                is_grasping = 0.

            assert len(objects_coordinates) == self.n_objects

            grasper_coordinates = np.asarray(grasper_coordinates)
            objects_coordinates = np.asarray(objects_coordinates)
            objects_is_grasped = np.asarray(objects_is_grasped)

            return self.get_entities_from_list_descriptions(grasper_coordinates, is_grasping,
                                                            objects_coordinates, objects_is_grasped)

        else:
            raise NotImplementedError

    def render(self, mode='human', everything=False):
        if not self.verbose:
            return
        img = self.encode_obs(self.entities, 'tile')
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        fig.suptitle(f'current goal to build : {self.goal}')
        img = img.swapaxes(0, 1)
        plt.imshow(img, origin='lower')  # we have to do this because imshow is row,column not x,y
        fig.canvas.draw()
        plt.pause(0.00001)
        if self.save_fig:
            plt.savefig(f'./figures/{self.i}')
            self.i += 1
        if everything:
            self.render_legend(mode)
        return img

    def _compute_n_steps_optim(self, obs):
        val, n_steps = self.heuristic_value(obs, discount=0., return_steps=True)
        return n_steps

    def render_legend(self, mode='human'):
        if not self.verbose:
            return
        legends = [[np.array([0., 0., 0.]), 'empty spot'],
                   [np.array([1., 0., 0.]), 'block on floor'],
                   [np.array([0., 1., 0.]), 'empty grasper on empty spot'],
                   [np.array([1., 1., 0.]), 'empty grasper on block on floor'],
                   [np.array([0., 1., 1.]), 'grasped block on empty spot'],
                   [np.array([1., 1., 1.]), 'grasped block on block on floor']]
        fig, axis = plt.subplots(len(legends))
        fig.suptitle('Color legend')

        for ax, leg in zip(axis, legends):
            im, text = leg
            im = im[None, None, :]
            ax.tick_params(left=False,
                           bottom=False,
                           labelleft=False,
                           labelbottom=False)
            ax.imshow(im)
            ax.set_title(text)
        fig.canvas.draw()
        plt.tight_layout()
        return


def render_image_on_figure(img, fig):
    fig.clear()
    plt.imshow(img.swapaxes(0, 1), origin='lower')  # we have to do this because imshow is row,column not x,y
    fig.canvas.draw()
    plt.pause(0.00001)
