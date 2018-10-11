"""
File Name: feature_extractor.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""


import math


class FeatureExtractor:

    def fuzzy_histog(self, x, y):
        '''
        This feature is calculated by dividing the normalized image grid in 4x4 cells which gives 25 corner points
        The membership value of a point in a cell for every corner is calculated and stored as a sum of that corner
        It returns a vector of length 25
        :param x: x coords of image
        :param y: y coords of image
        :return: 1x25 vector
        '''

        n_regions = 4
        x1 = 0
        x2 = 0.25
        x3 = 0
        x4 = 0.25

        y1 = 0
        y2 = 0
        y3 = 0.25
        y4 = 0.25

        res = {}

        for _ in range(n_regions):

            for _ in range(n_regions):
                res[(x1, y1)] = 0
                res[(x2, y2)] = 0
                res[(x3, y3)] = 0
                res[(x4, y4)] = 0

                x1 += 0.25
                x2 += 0.25
                x3 += 0.25
                x4 += 0.25

            x1 = 0
            x2 = 0.25
            x3 = 0
            x4 = 0.25

            y1 += 0.25
            y2 += 0.25
            y3 += 0.25
            y4 += 0.25

        x1 = 0
        x2 = 0.25
        x3 = 0
        x4 = 0.25

        y1 = 0
        y2 = 0
        y3 = 0.25
        y4 = 0.25

        for _ in range(n_regions):

            for _ in range(n_regions):

                i = 0
                while i < len(x):
                    if x[i] < x2 and y[i] < y4:
                        mp1 = ((0.25 - abs(x[i] - x1)) / 0.25) * ((0.25 - abs(y[i] - y1)) / 0.25)
                        mp2 = ((0.25 - abs(x[i] - x2)) / 0.25) * ((0.25 - abs(y[i] - y2)) / 0.25)
                        mp3 = ((0.25 - abs(x[i] - x3)) / 0.25) * ((0.25 - abs(y[i] - y3)) / 0.25)
                        mp4 = ((0.25 - abs(x[i] - x4)) / 0.25) * ((0.25 - abs(y[i] - y4)) / 0.25)

                        res[(x1, y1)] += mp1
                        res[(x2, y2)] += mp2
                        res[(x3, y3)] += mp3
                        res[(x4, y4)] += mp4
                    i += 1

                x1 += 0.25
                x2 += 0.25
                x3 += 0.25
                x4 += 0.25

            x1 = 0
            x2 = 0.25
            x3 = 0
            x4 = 0.25

            y1 += 0.25
            y2 += 0.25
            y3 += 0.25
            y4 += 0.25

        for key in res:
            res[key] /= len(x)

        features = []

        for key in sorted(res.keys()):
            features.append(res[key])

        return features

    def line_length(self, x, y):
        """
        Calculates lengh of line in the whole trace
        :param x: x coords of image
        :param y: y coords of image
        :return:
        """
        l = []
        l0 = 0
        l.append(l0)
        for i in range(1, len(x)):
            dist = math.pow(x[i] - x[i - 1], 2) + math.pow(y[i] - y[i - 1], 2)
            eucd = math.sqrt(dist)
            l0 = l0 + eucd
            l.append(l0)

        return l0

    def find_sharp(self, x, y):
        """
        Finds sharp edges in the image.
        :param x: x coords of image
        :param y: y coords of image
        :return:
        """

        i = 1
        theta = [0]
        new_x = [x[0]]
        new_y = [x[0]]
        while i < len(x) - 1:
            if (x[i] - x[i - 1]) == 0:
                a1 = (y[i] - y[i - 1])
            else:
                a1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1])

            if x[i + 1] - x[i] == 0:
                a2 = (y[i + 1] - y[i])
            else:
                a2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            theta_val = a2 - a1
            theta.append(theta_val)
            if theta_val == 0:
                i += 1
            else:
                delta = theta[i] - theta[i - 1]
                if delta <= 0 and theta[i - 1] != 0:
                    new_x.append(x[i])
                    new_y.append(x[i + 1])

                i += 1

        new_x.append(x[-1])
        new_y.append(y[-1])

        return new_x, new_y

    def crossing(self, x, y):
        """
        Divides x axis in 5 regions and y axis in 5 regions. Passes 9 straight lines from every region, sees if trace
        intersects these line, notes the starting and ending point of intersection and takes average. Creates 3 values
        for each region. Returns 10 vectors each containing 5 values for x and y coord each
        :param x: x coords of image
        :param y: y coords of image
        :return:
        """

        n_regions = 5
        x1 = 0
        x2 = 0.2
        y1 = 0
        y2 = 0.2
        region_vecx = []
        region_vecy = []
        position_fx = []
        position_lx = []
        position_fy = []
        position_ly = []

        position_fx_h = []
        position_lx_h = []
        position_fy_h = []
        position_ly_h = []

        for _ in range(n_regions):
            offset_x = 0.02
            x_points = []
            y_points = []
            intersectx = 0
            intersecty = 0
            nx = x1
            ny = y1

            for i in range(9):
                nx += offset_x
                x_points.append(nx)

            intpoint_x = []
            intpoint_y = []

            first_x = 0
            last_x = 0
            first_y = 0
            last_y = 0
            for val in x_points:
                i = 0
                while i < len(x):
                    if x[i] == val:
                        intersectx += 1
                        intpoint_x.append(x[i])
                        intpoint_y.append(y[i])
                    i += 1
                if len(intpoint_x) > 0:
                    first_x += intpoint_x[0]
                    last_x += intpoint_x[-1]
                    first_y += intpoint_y[0]
                    last_y += intpoint_y[-1]

            position_fx.append(first_x / 9)
            position_fy.append(first_y / 9)
            position_lx.append(last_x / 9)
            position_ly.append(last_y / 9)
            resx = intersectx / 9
            region_vecx.append(resx)

            x1 += 0.2
            x2 += 0.2

            yint_x = []
            yint_y = []
            yint_fx = 0
            yint_lx = 0
            yint_fy = 0
            yint_ly = 0
            for i in range(9):
                ny += offset_x
                y_points.append(ny)

            for val in y_points:
                i = 0
                while i < len(y):
                    if y[i] == val:
                        intersecty += 1
                        yint_x.append(x[i])
                        yint_y.append(y[i])
                    i += 1

                if len(yint_x) > 0:
                    yint_fx += yint_x[0]
                    yint_lx += yint_x[-1]
                    yint_fy += yint_y[0]
                    yint_ly += yint_y[-1]

            position_fx_h.append(yint_fx / 9)
            position_fy_h.append(yint_fy / 9)
            position_lx_h.append(yint_lx / 9)
            position_ly_h.append(yint_ly / 9)

            resy = intersecty / 9
            region_vecy.append(resy)

            y1 += 0.2
            y2 += 0.2

        return region_vecx, region_vecy, position_fx, position_fy, position_lx, position_ly, position_fx_h, position_fy_h, position_lx_h, position_ly_h

    def ndtse(self, xs, ys, xe, ye, x, y):
        """
        Calculates normalized distance to stroke edge

        :param xs: start x point of stroke
        :param ys: start y point of stroke
        :param xe: end x point of stroke
        :param ye: end y point of stroke
        :param x:  point x
        :param y:  point y
        :return:   ndtse actual and interpolated
        """

        ls = math.sqrt(((xs - xe) ** 2) + ((ys - ye) ** 2))
        db = math.sqrt(((xs - x) ** 2) + ((ys - y) ** 2))
        de = math.sqrt(((xe - x) ** 2) + ((ye - y) ** 2))

        if ls == 0:
            ls = 1
        ndtse_actual = 1 - (abs(de - db) / ls)
        ndtse_interpolated = -1 * ndtse_actual

        return ndtse_actual, ndtse_interpolated

    def curvature(self, x, y, x2, y2, x4, y4):
        """

        :param x: curr x
        :param y:  curr y
        :param x2: x-2
        :param y2: y-2
        :param x4:  x+2
        :param y4:  y+2
        :return:
        """
        # print("Division is " , x2,x,y2,y)
        if x2 == x:
            x2 = x2 - 0.00005

        if x4 == x:
            x4 = x4 - 0.00005

        slope1 = (y2 - y) / (x2 - x)
        b1 = y - (slope1 * x)

        if y < 0:
            a = [-slope1, -1]
        else:
            a = [-slope1, 1]

        slope2 = (y4 - y) / (x4 - x)
        b2 = y - (slope2 * x)

        if y < 0:
            b = [-slope2, -1]
        else:
            b = [-slope2, 1]

        da = math.sqrt((a[0] ** 2) + (a[1] ** 2))
        db = math.sqrt((b[0] ** 2) + (b[1] ** 2))

        cosq = ((a[0] * b[0]) + (a[1] * b[1])) / da * db

        return cosq

    def vicinity(self, x2, y2, x4, y4):
        """

        :param x2: x-2
        :param y2: y-2
        :param x4:  x+2
        :param y4:  y+2
        :return:
        """

        if x2 == x4:
            x2 = x2 - 0.00005

        slope1 = (y2 - y4) / (x2 - x4)
        b1 = y4 - (slope1 * x4)

        if y2 < 0:
            a = [-slope1, -1]
        else:
            a = [-slope1, 1]

        x5 = x2 + 10
        y5 = y2

        slope2 = (y5 - y2) / (x5 - x2)
        b2 = y5 - (slope2 * x5)

        if y2 < 0:
            b = [-slope2, -1]
        else:
            b = [-slope2, 1]

        da = math.sqrt((a[0] ** 2) + (a[1] ** 2))
        db = math.sqrt((b[0] ** 2) + (b[1] ** 2))

        cosq = ((a[0] * b[0]) + (a[1] * b[1])) / da * db
        sinq = math.sqrt(1 - (cosq) ** 2)

        return sinq

    def covariance_xy(self, x, y):
        """
        Calculates covariance of x and y
        :param x:
        :param x: x coords of image
        :param y: y coords of image
        """

        sum_y = sum(y)
        mean_x = sum_y / len(y)

        sum_x = sum(x)
        mean_y = sum_x / len(x)

        cov_xy = 0
        for i in range(len(x)):
            cov_xy = cov_xy + ((x[i] - mean_x) * (y[i] - mean_y))

        cov = cov_xy / len(x) - 1

        return cov
