import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath


class drawMask(object):

	def __init__(self, img):
		self.img = img
		self.fig = plt.gcf()
		self.ax = plt.gca()
		self.previous_point = None
		self.start_point = None
		self.end_point = None
		self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)
		self.fig.canvas.mpl_connect('button_press_event', self.button_press_event)
		plt.show()

	def button_press_event(self, event):
		if event.inaxes:
			x, y = event.xdata, event.ydata
			if event.button == 1:
				if self.start_point == None:
					self.start_point = [x,y]
					self.previous_point =  self.start_point
					self.x_point=[x]
					self.y_point=[y]   
					self.line = plt.Line2D([x, x],[y, y], marker = 'o')               
					self.ax.add_line(self.line)
					self.fig.canvas.draw()
				else:
					self.previous_point = [x,y]
					self.x_point.append(x)
					self.y_point.append(y)
					self.line = plt.Line2D([self.previous_point[0], x],[self.previous_point[1], y],marker = 'o')
					event.inaxes.add_line(self.line)
					self.fig.canvas.draw()
			elif (event.button == 3):
				self.line.set_data([self.previous_point[0],self.start_point[0]],
								   [self.previous_point[1],self.start_point[1]])
				self.ax.add_line(self.line)
				self.fig.canvas.draw()
				self.line = None
				plt.close(self.fig)

	def motion_notify_event(self, event):
		if event.inaxes:
			x, y = event.xdata, event.ydata
			if (event.button == None) and self.start_point != None:
				self.line.set_data([self.previous_point[0], x],
									[self.previous_point[1], y])
				self.fig.canvas.draw()

	def draw_mask(self):
		img_h = self.img.shape[0]
		img_w = self.img.shape[1]
		mesh_x, mesh_y = np.meshgrid(np.arange(img_w),np.arange(img_h))
		point = np.zeros((mesh_x.size,2))
		point[:,0] = mesh_x.flatten()
		point[:,1] = mesh_y.flatten()

		poly_point = np.zeros((len(self.x_point),2))

		poly_point[:,0] = np.array(self.x_point)
		poly_point[:,1] = np.array(self.y_point)
		poly_path = mplPath.Path(poly_point)
		poly_mask = poly_path.contains_points(point).reshape(img_h,img_w)
		return poly_mask

	def get_bbox(self, mask):
		coor = np.argwhere(mask)
		h_min = np.min(coor[:,0])
		h_max = np.max(coor[:,0])
		w_min = np.min(coor[:,1])
		w_max = np.max(coor[:,1])

		return [w_min, h_min, w_max, h_max]



def draw_mask(img):
	plt.imshow(img)
	draw = drawMask(img)
	mask = draw.draw_mask()
	bbox = draw.get_bbox(mask)
	plt.imshow(mask)
	plt.show()

	return mask.astype(int) , bbox

