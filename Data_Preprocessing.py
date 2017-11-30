import dicom as dc
import numpy as np
import os
import fnmatch
from random import randint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from xml.etree import ElementTree
from shapely.geometry import Polygon
from shapely.geometry import Point

# set up the training data file system
class Data_Preprocessing(object):

    def create_folder(self, path):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                pass

    def find(self, pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name).replace("\\", "/"))
        return result

    def get_random_path(self):
        j = 0
        path = ''
        xml_path = ''
        while j<1000:
            path = self.training_list[randint(0, self.training_list_length-1)]
            k = -1
            while (not path[k] == '/') and k > -1000:
                k = k - 1
            last_number = len(path) + k
            cut_path = path[0:last_number]
            xml_path_list = self.find('*xml', cut_path)
            if xml_path_list:
                if self.valid_xml(xml_path_list[0]):
                    xml_path = xml_path_list[0]
                    j = 1000
            j = j+1
        if j == 1000:
            print('No legit data found')

        return path, xml_path

    # checks if the xml file is a valid source (has readingSessions instead of CXRreadingSessions)
    def valid_xml(self, xml_path):
        valid = True
        f = ElementTree.parse(xml_path).getroot()
        session = f.findall('{http://www.nih.gov}readingSession')
        if not session:
            valid = False
        return valid



    # gets and processes the data
    def get_data(self, path, xml_path):
        # find the slice of the given image
        dc_file = dc.read_file(path)
        z_position = float((dc_file[0x0020, 0x0032].value)[2])
        size = (dc_file.pixel_array).shape

        # read out xml annotation file
        f = ElementTree.parse(xml_path).getroot()
        annotation_list = []
        nodule_list = []
        for child in f.findall('{http://www.nih.gov}readingSession'):
            # the annotation mask of this radiologist
            annotations = np.zeros(shape=size)
            nodules = np.zeros(shape=size)
            # loop over nodules
            for grandchild in child.findall('{http://www.nih.gov}unblindedReadNodule'):
                # loop over 2-dim slices of a single nodule
                for ggc in grandchild.findall('{http://www.nih.gov}roi'):
                    image_z = float(ggc[0].text)
                    # check if current slice has correct z coordinate
                    if image_z == z_position:
                        print('Matching nodule found')
                        vertices = []
                        for coord in ggc.findall('{http://www.nih.gov}edgeMap'):
                            vertices.append((int(coord[0].text), int(coord[1].text)))
                            annotations[int(coord[0].text), int(coord[1].text)] = 1
                        poly = Polygon(vertices)
                        bnd = poly.bounds
                        for x in range(int(bnd[0]), int(bnd[2] + 1)):
                            for y in range(int(bnd[1]), int(bnd[3] + 1)):
                                point = Point(x, y)
                                if point.within(poly):
                                    nodules[x, y] = 1
            annotation_list.append(annotations)
            nodule_list.append(nodules)

        # renormalize pic
        pic = dc_file.pixel_array
        pic = pic - np.amin(pic)
        pic = pic / np.amax(pic)
        return pic, annotation_list, nodule_list

    # visualizes the nodule as black and white with red annotations
    def visualize_nodules(self, pic, nod, k):
        size = pic.shape
        three_c = np.zeros(shape=[size[0], size[1], 3])
        for k in range(3):
            three_c[..., k] = pic

        # set red channel to 1 whenever in nodules and set all other channels to 0
        for x in range(size[0]):
            for y in range(size[1]):
                if (nod)[x, y] == 1:
                    three_c[x, y, 0] = 1
                    three_c[x, y, 1] = 0
                    three_c[x, y, 2] = 0

        three_c[..., 0] = three_c[..., 0] + nod
        plt.figure()
        plt.imshow(three_c)
        plt.savefig('Data/Test/' + str(k) + '.jpg')
        plt.close()

    def visualize(self, image, k):
        plt.figure()
        plt.imshow(image)
        self.create_folder('Data/Test/')
        plt.savefig('Data/Test/' + str(k) + '.jpg')
        plt.close()

    def __init__(self):
        self.training_list = self.find('*.dcm', './Segmentation_Data')
        self.training_list_length = len(self.training_list)
        if self.training_list_length == 0:
            print('No training Data found')

