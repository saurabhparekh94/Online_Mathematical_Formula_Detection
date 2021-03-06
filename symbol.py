"""
File Name: symbol.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""

import math


class Symbol:
    """
    Create a symbol object
    functions include calculating geometric features
    """

    def __init__(self, x, y, label, stroke_id=None, sym_ct=0):
        self.x = x
        self.y = y
        self.symbol = label
        self.stroke_id = stroke_id
        self.sym_ct = sym_ct

    def __str__(self):
        return str(self.symbol) + "\n" + "X coord\n" + str(self.x) + "\n Y coord\n" + str(
            self.y) + "\n Trace ID " + str(self.stroke_id)

    def __len__(self):
        return len(self.x)

    def find_slope(self, y2, y1, x2, x1):
        if x2 == x1:
            slope = 9999
        else:
            slope = (y2 - y1) / (x2 - x1)

        return slope

    def writing_slope(self, next_symbol):
        '''
        The function finds the writing slope between two symbols
        :param next_symbol:
        :return: writing slope
        '''

        current_symbol_x = self.x
        next_symbol_x = next_symbol.x

        current_symbol_y = self.y
        next_symbol_y = next_symbol.y

        last_stroke_current_x = current_symbol_x[-1]
        first_stroke_next_x = next_symbol_x[0]

        last_point_current_x = last_stroke_current_x[-1]
        first_point_next_x = first_stroke_next_x[0]

        last_stroke_current_y = current_symbol_y[-1]
        first_stroke_next_y = next_symbol_y[0]

        last_point_current_y = last_stroke_current_y[-1]
        first_point_next_y = first_stroke_next_y[0]

        slope = self.find_slope(first_point_next_y, last_point_current_y, first_point_next_x, last_point_current_x)

        angle = math.atan(slope)

        return angle

    def writing_curvature(self, next_symbol):

        '''
        the symbols finds the writing curvature between two symbols
        :param next_symbol: the succesuve symbol
        :return: writing curvature angle
        '''

        strokes_x = self.x
        first_currentstroke_x = strokes_x[0]
        first_laststroke_x = strokes_x[-1]
        first_currentpoint_x = first_currentstroke_x[0]
        last_currentpoint_x = first_laststroke_x[-1]

        strokes_y = self.y
        first_currentstroke_y = strokes_y[0]
        first_laststroke_y = strokes_y[-1]
        first_currentpoint_y = first_currentstroke_y[0]
        last_currentpoint_y = first_laststroke_y[-1]

        strokes_next_x = next_symbol.x
        first_nextstroke_x = strokes_next_x[0]
        first_nextlaststroke_x = strokes_next_x[-1]
        first_nextpoint_x = first_nextstroke_x[0]
        last_nextpoint_x = first_nextlaststroke_x[-1]

        strokes_next_y = next_symbol.y
        first_nextstroke_y = strokes_next_y[0]
        first_nextlaststroke_y = strokes_next_y[-1]
        first_nextpoint_y = first_nextstroke_y[0]
        last_nextpoint_y = first_nextlaststroke_y[-1]

        m1 = self.find_slope(last_currentpoint_y, first_currentpoint_y, last_currentpoint_x, first_currentpoint_x)
        m2 = self.find_slope(last_nextpoint_y, first_nextpoint_y, last_nextpoint_x, first_nextpoint_x)
        if (1 + (m1 * m2)) == 0:
            tantheta = 90
        else:
            tantheta = (m1 - m2) / (1 + (m1 * m2))
        writingCurvatureAngle = math.atan(tantheta)

        return (writingCurvatureAngle)

    def join_stuff(self, x, y):
        '''
        :param x:x co-ordinates it a list of list
        :param y: y co-ordinates its a list of list
        :return: it returns 2 list, a single list of all merged x points and single list of all merged y points
        '''

        all_x = []
        all_y = []
        for items in range(len(x)):
            all_x = all_x + x[items]
            all_y = all_y + y[items]

        return all_x, all_y

    def find_MinMax(self, x):
        '''
        :param x: list whose minimum and maximumneeds to be find
        :return:minimum  and maximum point of a list
        '''

        mini = min(x)
        maxi = max(x)
        return mini, maxi

    def find_bounding_box_centre(self, min_x, max_x, min_y, max_y):
        '''
        :param min_x: minimum point on the bounding box
        :param max_x: max x point on the bounding box
        :param min_y: min y point on the bounding box
        :param max_y: max y point in the bounding point
        :return: centre of bounding box
        '''

        centre_x = (min_x + max_x) / 2
        centre_y = (min_y + max_y) / 2

        return centre_x, centre_y

    def distance_between_box(self, next_symbol):
        '''
        function finds distance between succesive symbol
        :param next_symbol: The succesive symbol
        :return: distance between bounding box
        '''

        x = self.x
        y = self.y

        current_merge_x, current_merge_y = self.join_stuff(x, y)

        new_x = next_symbol.x
        new_y = next_symbol.y

        next_merge_x, next_merge_y = self.join_stuff(new_x, new_y)

        current_min_x, current_max_x = self.find_MinMax(current_merge_x)
        current_min_y, current_max_y = self.find_MinMax(current_merge_y)

        current_centre_x, current_centre_y = self.find_bounding_box_centre(current_min_x, current_max_x, current_min_y,
                                                                           current_max_y)

        next_min_x, next_max_x = self.find_MinMax(next_merge_x)
        next_min_y, next_max_y = self.find_MinMax(next_merge_y)

        next_centre_x, next_centre_y = self.find_bounding_box_centre(next_min_x, next_max_x, next_min_y,
                                                                     next_max_y)

        distance_between_centres = math.sqrt(
            math.pow((current_centre_x - next_centre_x), 2) + math.pow((current_centre_y - next_centre_y), 2))

        return (distance_between_centres)

    def distance_between_average_centres(self, next_symbol):
        '''
        :param next_symbol: distance
        :return: distance between average centres, horizontal off set and vertical distance
        '''
        current_x = self.x
        current_y = self.y

        current_merge_x, current_merge_y = self.join_stuff(current_x, current_y)
        current_centre_x = sum(current_merge_x) / len(current_merge_x)
        current_centre_y = sum(current_merge_y) / len(current_merge_y)

        next_x = next_symbol.x
        next_y = next_symbol.y

        next_merge_x, next_merge_y = self.join_stuff(next_x, next_y)
        next_centre_x = sum(next_merge_x) / len(next_merge_x)
        next_centre_y = sum(next_merge_y) / len(next_merge_y)

        sum_of_square = math.pow((current_centre_x - next_centre_x), 2) + math.pow((current_centre_y - next_centre_y),
                                                                                   2)
        distance_centre_of_mass = math.sqrt(sum_of_square)

        horizontal_distance = next_centre_x - current_centre_x
        vertical_distance = next_centre_y - current_centre_y

        return distance_centre_of_mass, horizontal_distance, vertical_distance

    def maximal_point_distance(self, next_symbol):

        x = self.x
        y = self.y

        next_x = next_symbol.x
        next_y = next_symbol.y

        current_merge_x, current_merge_y = self.join_stuff(x, y)

        next_merge_x, next_merge_y = self.join_stuff(next_x, next_y)
        max = 0

        for i in range(len(current_merge_x) - 1):
            for j in range(i, len(next_merge_x)):
                y2 = next_merge_y[j]
                y1 = current_merge_y[i]
                x2 = next_merge_x[j]
                x1 = next_merge_x[i]
                sum_of_square = math.pow((x2 - x1), 2) + math.pow(
                    (y2 - y1), 2)
                distance = math.sqrt(sum_of_square)

                if max < distance:
                    max = distance

        return max
