"""
File Name: preprocessing.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""

import math

import numpy as np


class Preprocessing:

    def get_aspect(self, x, y):
        """
        returns aspect ratio of image
        :param x: x coords of image
        :param y: y coords of image
        :return:
        """

        y_min = min(y)
        y_max = max(y)
        x_max = max(x)
        x_min = min(x)

        height = math.sqrt((y_max - y_min) ** 2)
        width = math.sqrt((x_max - x_min) ** 2)

        if height == 0:
            height = 1

        ar = abs(width / height)

        return ar

    def _normalize(self, x, y, x_min, y_min, x_max, y_max):
        """
        Helper function of normalizing

        """

        if len(x) < 2:
            return x, y

        new_x = []
        new_y = []

        for val in x:
            diffx = (x_max - x_min)

            if diffx == 0:
                diffx = 1

            tx = (val - x_min) / diffx
            new_x.append(tx)

        for val in y:
            diffy = (y_max - y_min)
            if diffy == 0:
                diffy = 1

            ty = (val - y_min) / diffy
            new_y.append(ty)

        return new_x, new_y

    def normalize(self, x, y, parser=False):
        """
        Normalizes x and y between 0 and 1
        :param x: x coords of image
        :param y: y coords of image
        :return:
        """

        if parser:

            x_max = -10000
            for l in x:
                for p in l:
                    if p > x_max:
                        x_max = p
            y_max = -10000
            for l in y:
                for p in l:
                    if p > y_max:
                        y_max = p

            x_min = 100000
            for l in x:
                for p in l:
                    if p < x_min:
                        x_min = p

            y_min = 100000
            for l in y:
                for p in l:
                    if p < y_min:
                        y_min = p

            if x_max == x_min and y_max == y_min:
                return x, y

            for i in range(len(x)):
                x[i], y[i] = self._normalize(x[i], y[i], x_min, y_min, x_max, y_max)

            return x, y
        else:
            y_min = min(y)
            y_max = max(y)
            x_max = max(x)
            x_min = min(x)

            x, y = self._normalize(x, y, x_min, y_min, x_max, y_max)

            return x, y

    def _size_norm(self, x, y):
        """
        Helper function of size normalization
        """

        new_width = self.get_aspect(x, y)

        y_max = max(y)
        y_min = min(y)
        x_min = min(x)

        new_x, new_y = [], []

        for val in y:
            diffy = (y_max - y_min)
            if diffy == 0:
                diffy = 1
            ty = (val - y_min) / diffy
            new_y.append(ty)

        x_max = x_min + new_width

        for val in x:
            diffx = (max(x) - min(x))
            b = x_max
            a = x_min
            if diffx == 0 :
                diffx = 1
            tx = ((b-a)*((val - x_min)/diffx)) + a
            new_x.append(tx)


        return new_x, new_y


    def size_norm(self, x, y, parser=False):

        if parser:

            for i in range(len(x)):
                x[i], y[i] = self._size_norm(x[i], y[i])

        else:
            x, y = self._size_norm(x, y)

        return x, y

    def _duplicate_filter(self, x, y):
        """
        Helper function for duplicate filter
        """

        new_x = []
        new_y = []

        new_x.append(x[0])
        new_y.append(y[0])
        for items in range(len(x) - 1):
            if x[items + 1] == x[items] and y[items + 1] == y[items]:
                pass
            else:
                new_x.append(x[items + 1])
                new_y.append(y[items + 1])

        return new_x, new_y

    def duplicate_filter(self, x, y, parser=False):
        """
        Removes duplicate points from Trace
        :param x: x coords of image
        :param y: y coords of image
        :return:
        """

        if parser:
            for i in range(len(x)):
                x[i], y[i] = self._duplicate_filter(x[i], y[i])
        else:
            x, y = self._duplicate_filter(x, y)

        return x, y

    def _smoothen(self, x, y):
        """
        Helper function for smoothening
        """

        i = 1
        while i < len(x) - 1:
            # smooth x
            new_x = (x[i - 1] + x[i] + x[i + 1]) / 3
            x[i] = new_x

            # smooth y
            new_y = (y[i - 1] + y[i] + y[i + 1]) / 3
            y[i] = new_y

            i += 1

        return x, y

    def smoothen(self, x, y, parser=False):
        """
        Smoothens the values of x and y
        Formula : (x[i - 1] + x[i] + x[i + 1]) / 3
        :param x:
        :param y:
        :return:
        """
        if parser:
            for i in range(len(x)):
                x[i], y[i] = self._smoothen(x[i], y[i])
        else:
            x, y = self._smoothen(x, y)

        return x, y

    def dopreprocess(self, traces=None, symbol=True, x=None, y=None, parser=False):
        """
        Function that calls preprocessing as required
        """

        if parser:
            x, y = self.duplicate_filter(x, y, parser)
            x, y = self.smoothen(x, y, parser)
            x, y = self.size_norm(x, y, parser)
            x, y = self.duplicate_filter(x, y, parser)
            return x, y
        elif symbol:
            x, y = self.duplicate_filter(x, y)
            x, y = self.normalize(x, y)
            x, y = self.smoothen(x, y)
            x, y = self.duplicate_filter(x, y)
            x = np.around(x, decimals=2)
            y = np.around(y, decimals=2)
            return x, y
        else:
            for key in traces:
                trace = traces[key]
                trace['x'], trace['y'] = self.duplicate_filter(trace['x'], trace['y'])
                trace['x'], trace['y'] = self.smoothen(trace['x'], trace['y'])
                trace['x'], trace['y'] = self.size_norm(trace['x'], trace['y'])
                trace['x'], trace['y'] = self.duplicate_filter(trace['x'], trace['y'])
            return traces
