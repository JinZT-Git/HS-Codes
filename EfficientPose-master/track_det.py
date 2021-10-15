# -*- coding: utf-8 -*-
# @Time    : 2021/6/11 20:52
# @Author  : JZT
# @Email   : 915681919@qq.com
# @File    : track_det.py
# @Software: PyCharm

from os.path import join, normpath
from pymediainfo import MediaInfo
import numpy as np
import time
import glob
from utils import helpers

"""
	读取检测的识别结果，框选单个人的矩形，分别进行姿态识别，并将结果映射到原图大小
"""

f = open(r"E:\2.Pythonfile\yolov5-develop\path_and_pos.txt", "r")
txt_name_list = {}
for line in f:
	line_list = line[:-1].split("*")
	pos_list = []
	for i in range((len(line_list)-1)//4):
		x1, y1, x2, y2 = int(line_list[i*4+1]), int(line_list[i*4+2]), int(line_list[i*4+3]), int(line_list[i*4+4])
		pos_list.append([x1, y1, x2, y2])
	txt_name_list[line_list[0]] = pos_list
f.close()
print("txt中图片数量 ：", len(txt_name_list))


def get_model(framework, model_variant):
	"""
	Load the desired EfficientPose model variant using the requested deep learning framework.

	Args:
		framework: string
			Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
		model_variant: string
			EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)

	Returns:
		Initialized EfficientPose model and corresponding resolution.
	"""
	
	# Keras
	if framework in ['keras', 'k']:
		from tensorflow.keras.backend import set_learning_phase
		from tensorflow.keras.models import load_model
		set_learning_phase(0)
		model = load_model(join('models', 'keras', 'EfficientPose{0}.h5'.format(model_variant.upper())),
		                   custom_objects={'BilinearWeights': helpers.keras_BilinearWeights, 'Swish': helpers.Swish(helpers.eswish), 'eswish': helpers.eswish,
		                                   'swish1': helpers.swish1})
	
	# TensorFlow
	elif framework in ['tensorflow', 'tf']:
		from tensorflow.python.platform.gfile import FastGFile
		from tensorflow.compat.v1 import GraphDef
		from tensorflow.compat.v1.keras.backend import get_session
		from tensorflow import import_graph_def
		f = FastGFile(join('models', 'tensorflow', 'EfficientPose{0}.pb'.format(model_variant.upper())), 'rb')
		graph_def = GraphDef()
		graph_def.ParseFromString(f.read())
		f.close()
		model = get_session()
		model.graph.as_default()
		import_graph_def(graph_def)
	
	# TensorFlow Lite
	elif framework in ['tensorflowlite', 'tflite']:
		from tensorflow import lite
		model = lite.Interpreter(model_path=join('models', 'tflite', 'EfficientPose{0}.tflite'.format(model_variant.upper())))
		model.allocate_tensors()
	
	# PyTorch
	elif framework in ['pytorch', 'torch']:
		from imp import load_source
		from torch import load, quantization, backends
		try:
			MainModel = load_source('MainModel', join('models', 'pytorch', 'EfficientPose{0}.py'.format(model_variant.upper())))
		except:
			print('\n##########################################################################################################')
			print(
				'Desired model "EfficientPose{0}Lite" not available in PyTorch. Please select among "RT", "I", "II", "III" or "IV".'.format(model_variant.split('lite')[0].upper()))
			print('##########################################################################################################\n')
			return False, False
		model = load(join('models', 'pytorch', 'EfficientPose{0}'.format(model_variant.upper())))
		model.eval()
		qconfig = quantization.get_default_qconfig('qnnpack')
		backends.quantized.engine = 'qnnpack'
	
	return model, {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}[model_variant]


def infer(batch, model, lite, framework):
	"""
	Perform inference on supplied image batch.

	Args:
		batch: ndarray
			Stack of preprocessed images
		model: deep learning model
			Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
		lite: boolean
			Defines if EfficientPose Lite model is used
		framework: string
			Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)

	Returns:
		EfficientPose model outputs for the supplied batch.
	"""
	
	# Keras
	if framework in ['keras', 'k']:
		if lite:
			batch_outputs = model.predict(batch)
		else:
			batch_outputs = model.predict(batch)[-1]
	
	# TensorFlow
	elif framework in ['tensorflow', 'tf']:
		output_tensor = model.graph.get_tensor_by_name('upscaled_confs/BiasAdd:0')
		if lite:
			batch_outputs = model.run(output_tensor, {'input_1_0:0': batch})
		else:
			batch_outputs = model.run(output_tensor, {'input_res1:0': batch})
	
	# TensorFlow Lite
	elif framework in ['tensorflowlite', 'tflite']:
		input_details = model.get_input_details()
		output_details = model.get_output_details()
		model.set_tensor(input_details[0]['index'], batch)
		model.invoke()
		batch_outputs = model.get_tensor(output_details[-1]['index'])
	
	# PyTorch
	elif framework in ['pytorch', 'torch']:
		from torch import from_numpy, autograd
		batch = np.rollaxis(batch, 3, 1)
		batch = from_numpy(batch)
		batch = autograd.Variable(batch, requires_grad=False).float()
		batch_outputs = model(batch)
		batch_outputs = batch_outputs.detach().numpy()
		batch_outputs = np.rollaxis(batch_outputs, 1, 4)
	
	return batch_outputs


def analyze_camera(model, framework, resolution, lite):
	"""
	Live prediction of pose coordinates from camera.

	Args:
		model: deep learning model
			Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
		framework: string
			Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
		resolution: int
			Input height and width of model to utilize
		lite: boolean
			Defines if EfficientPose Lite model is used

	Returns:
		Predicted pose coordinates in all frames of camera session.
	"""
	
	# Load video
	import cv2
	start_time = time.time()
	cap = cv2.VideoCapture(0)
	_, frame = cap.read()
	frame_height, frame_width = frame.shape[:2]
	coordinates = []
	print('\n##########################################################################################################')
	while (True):
		
		# Read frame
		_, frame = cap.read()
		
		# Construct batch
		batch = [frame[..., ::-1]]
		
		# Preprocess batch
		batch = helpers.preprocess(batch, resolution, lite)
		
		# Perform inference
		batch_outputs = infer(batch, model, lite, framework)
		
		# Extract coordinates for frame
		frame_coordinates = helpers.extract_coordinates(batch_outputs[0, ...], frame_height, frame_width, real_time=True)
		coordinates += [frame_coordinates]
		
		# Draw and display predictions
		helpers.display_camera(cv2, frame, frame_coordinates, frame_height, frame_width)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()
	
	# Print total operation time
	print('Camera operated in {0} seconds'.format(time.time() - start_time))
	print('##########################################################################################################\n')
	
	return coordinates


def analyze_image(file_path, model, framework, resolution, lite):
	"""
	Predict pose coordinates on supplied image.

	Args:
		file_path: path
			System path of image to analyze
		model: deep learning model
			Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
		framework: string
			Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
		resolution: int
			Input height and width of model to utilize
		lite: boolean
			Defines if EfficientPose Lite model is used

	Returns:
		Predicted pose coordinates in the supplied image.
	"""
	
	# Load image
	from PIL import Image
	
	start_time = time.time()

	All_img = Image.open(file_path)
	W, H = All_img.size
	
	pos_list = txt_name_list[file_path]
	coordinates_list = []
	
	for pos_one in pos_list:
		
		image = np.array(All_img.crop((pos_one[0], pos_one[1], pos_one[2], pos_one[3])))
		
		
		# print(pos_one[0], pos_one[1], pos_one[2], pos_one[3])
		
		dx = pos_one[2] - pos_one[0]
		dy = pos_one[3] - pos_one[1]
		
		image_height, image_width = image.shape[:2]
		
		# print(image_height, image_width)
		# exit()
		batch = np.expand_dims(image, axis=0)
		
		# Preprocess batch
		batch = helpers.preprocess(batch, resolution, lite)
		
		# Perform inference
		batch_outputs = infer(batch, model, lite, framework)
		
		# Extract coordinates
		coordinates = [helpers.extract_coordinates(batch_outputs[0, ...], image_height, image_width)]
		
		for ii in range(len(coordinates[0])):
			
			tmp_tuple_list = list(coordinates[0][ii])
			
			tmp_tuple_list[1] = (tmp_tuple_list[1] * image_width + pos_one[0]) / W
			tmp_tuple_list[2] = (tmp_tuple_list[2] * image_height + pos_one[1]) / H
			
			coordinates[0][ii] = tuple(tmp_tuple_list)

		coordinates_list.append(coordinates)
	# Print processing time
	print('\n##########################################################################################################')
	print('Image processed in {0} seconds'.format('%.3f' % (time.time() - start_time)))
	print('##########################################################################################################\n')
	return coordinates_list


def analyze_video(file_path, model, framework, resolution, lite):
	"""
	Predict pose coordinates on supplied video.

	Args:
		file_path: path
			System path of video to analyze
		model: deep learning model
			Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
		framework: string
			Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
		resolution: int
			Input height and width of model to utilize
		lite: boolean
			Defines if EfficientPose Lite model is used

	Returns:
		Predicted pose coordinates in all frames of the supplied video.
	"""
	
	# Define batch size and number of batches in each part
	batch_size = 1 if framework in ['tensorflowlite', 'tflite'] else 49
	part_size = 490 if framework in ['tensorflowlite', 'tflite'] else 10
	
	# Load video
	from skvideo.io import vreader, ffprobe
	start_time = time.time()
	try:
		videogen = vreader(file_path)
		video_metadata = ffprobe(file_path)['video']
		num_video_frames = int(video_metadata['@nb_frames'])
		num_batches = int(np.ceil(num_video_frames / batch_size))
		frame_height, frame_width = next(vreader(file_path)).shape[:2]
	except:
		print('\n##########################################################################################################')
		print('Video "{0}" could not be loaded. Please verify that the file is working.'.format(file_path))
		print('##########################################################################################################\n')
		return False
	
	# Operate on batches
	coordinates = []
	batch_num = 1
	part_start_time = time.time()
	print('\n##########################################################################################################')
	while True:
		
		# Fetch batch of frames
		batch = [next(videogen, None) for _ in range(batch_size)]
		if not type(batch[0]) == np.ndarray:
			break
		elif not type(batch[-1]) == np.ndarray:
			batch = [frame if type(frame) == np.ndarray else np.zeros((frame_height, frame_width, 3)) for frame in batch]
		
		# Preprocess batch
		batch = helpers.preprocess(batch, resolution, lite)
		
		# Perform inference
		batch_outputs = infer(batch, model, lite, framework)
		
		# Extract coordinates for batch
		batch_coordinates = [helpers.extract_coordinates(batch_outputs[n, ...], frame_height, frame_width) for n in range(batch_size)]
		coordinates += batch_coordinates
		
		# Print partial processing time
		if batch_num % part_size == 0:
			print('{0} of {1}: Part processed in {2} seconds | Video processed for {3} seconds'.format(int(batch_num / part_size), int(np.ceil(num_batches / part_size)),
			                                                                                           '%.3f' % (time.time() - part_start_time),
			                                                                                           '%.3f' % (time.time() - start_time)))
			part_start_time = time.time()
		batch_num += 1
	
	# Print total processing time
	print('{0} of {0}: Video processed in {1} seconds'.format(int(np.ceil(num_batches / part_size)), '%.3f' % (time.time() - start_time)))
	print('##########################################################################################################\n')
	
	return coordinates[:num_video_frames]


def analyze(video, file_path, model, framework, resolution, lite):
	"""
	Analyzes supplied camera/video/image.

	Args:
		video: boolean
			Flag if video is supplied, else assumes image
		file_path: path
			System path of video/image to analyze, None for camera
		model: deep learning model
			Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
		framework: string
			Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
		resolution: int
			Input height and width of model to utilize
		lite: boolean
			Defines if EfficientPose Lite model is used

	Returns:
		Predicted pose coordinates in supplied video/image.
	"""
	
	# Camera-based analysis
	if file_path is None:
		coordinates = analyze_camera(model, framework, resolution, lite)
	
	# Video analysis
	elif video:
		coordinates = analyze_video(file_path, model, framework, resolution, lite)
	
	# Image analysis
	else:
		coordinates = analyze_image(file_path, model, framework, resolution, lite)
	
	return coordinates


def annotate_image(file_path, coordinates):
	"""
	Annotates supplied image from predicted coordinates.

	Args:
		file_path: path
			System path of image to annotate
		coordinates: list
			Predicted body part coordinates for image
	"""
	
	# Load raw image
	from PIL import Image, ImageDraw
	image = Image.open(file_path)
	image_width, image_height = image.size
	print(file_path)
	print(image_width, image_height)
	
	image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
	
	if len(coordinates) < 1:
		print("error NO pos in ", file_path)
	else:
		for coordinates_one in coordinates:
			image_width, image_height = image.size
			image_side = image_width if image_width >= image_height else image_height
			
			# Annotate image
			image_draw = ImageDraw.Draw(image)
			image_coordinates = coordinates_one[0]
			image = helpers.display_body_parts(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, marker_radius=int(image_side / 75))
			image = helpers.display_segments(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, segment_width=int(image_side / 50))
		
	# Save annotated image
	image.save(normpath((file_path[:-4] + '_tracked.jpg').replace("gldw_zt", "gldw_zt_pos")))


def annotate_video(file_path, coordinates):
	"""
	Annotates supplied video from predicted coordinates.

	Args:
		file_path: path
			System path of video to annotate
		coordinates: list
			Predicted body part coordinates for each frame in the video
	"""
	
	# Load raw video
	from skvideo.io import vreader, ffprobe, FFmpegWriter
	videogen = vreader(file_path)
	video_metadata = ffprobe(file_path)['video']
	fps = video_metadata['@r_frame_rate']
	frame_height, frame_width = next(vreader(file_path)).shape[:2]
	frame_side = frame_width if frame_width >= frame_height else frame_height
	
	# Initialize annotated video
	vcodec = 'libvpx-vp9'  # 'libx264'
	writer = FFmpegWriter(normpath(file_path.split('.')[0] + '_tracked.mp4'), inputdict={'-r': fps},
	                      outputdict={'-r': fps, '-bitrate': '-1', '-vcodec': vcodec, '-pix_fmt': 'yuv420p', '-lossless': '1'})  # '-lossless': '1'
	
	# Annotate video
	from PIL import Image, ImageDraw
	i = 0
	while True:
		try:
			frame = next(videogen)
			image = Image.fromarray(frame)
			image_draw = ImageDraw.Draw(image)
			image_coordinates = coordinates[i]
			image = helpers.display_body_parts(image, image_draw, image_coordinates, image_height=frame_height, image_width=frame_width, marker_radius=int(frame_side / 150))
			image = helpers.display_segments(image, image_draw, image_coordinates, image_height=frame_height, image_width=frame_width, segment_width=int(frame_side / 100))
			writer.writeFrame(np.array(image))
			i += 1
		except:
			break
	
	writer.close()


def annotate(video, file_path, coordinates):
	"""
	Analyzes supplied video/image from predicted coordinates.

	Args:
		video: boolean
			Flag if video is supplied, else assumes image
		file_path: path
			System path of video/image to annotate
		coordinates: list
			Predicted body part coordinates for video/image
	"""
	
	# Annotate video predictions
	if video:
		coordinates = annotate_video(file_path, coordinates)
	
	# Annotate image predictions
	else:
		coordinates = annotate_image(file_path, coordinates)


def save(video, file_path, coordinates):
	"""
	Saves predicted coordinates as CSV.

	Args:
		video: boolean
			Flag if video is supplied, else assumes image
		file_path: path
			System path of video/image to annotate
		coordinates: list
			Predicted body part coordinates for video/image
	"""
	
	# Initialize CSV
	import csv
	csv_path = normpath(file_path.split('.')[0] + '_coordinates.csv') if file_path is not None else normpath('camera_coordinates.csv')
	csv_file = open(csv_path, 'w')
	headers = ['frame'] if video else []
	[headers.extend([body_part + '_x', body_part + '_y']) for body_part, _, _ in coordinates[0]]
	writer = csv.DictWriter(csv_file, fieldnames=headers)
	writer.writeheader()
	
	# Write coordinates to CSV
	for i, image_coordinates in enumerate(coordinates):
		row = {'frame': i + 1} if video else {}
		for body_part, body_part_x, body_part_y in image_coordinates:
			row[body_part + '_x'] = body_part_x
			row[body_part + '_y'] = body_part_y
		writer.writerow(row)
	
	csv_file.flush()
	csv_file.close()


def perform_tracking(video, file_path, model_name, framework_name, visualize, store):
	"""
	Process of estimating poses from camera/video/image.

	Args:
		video: boolean
			Flag if camera or video is supplied, else assumes image
		file_path: path
			System path of video/image to analyze, None if camera
		model_name: string
			EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
		framework_name: string
			Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
		visualize: boolean
			Flag to create visualization of predicted poses
		store: boolean
			Flag to create CSV file with predicted coordinates

	Returns:
		Boolean expressing if tracking was successfully performed.
	"""
	
	# VERIFY FRAMEWORK AND MODEL VARIANT
	framework = framework_name.lower()
	model_variant = model_name.lower()
	if framework not in ['keras', 'k', 'tensorflow', 'tf', 'tensorflowlite', 'tflite', 'pytorch', 'torch']:
		print('\n##########################################################################################################')
		print('Desired framework "{0}" not available. Please select among "tflite", "tensorflow", "keras" or "pytorch".'.format(framework_name))
		print('##########################################################################################################\n')
		return False
	elif model_variant not in ['efficientposert', 'rt', 'efficientposei', 'i', 'efficientposeii', 'ii', 'efficientposeiii', 'iii', 'efficientposeiv', 'iv', 'efficientposert_lite',
	                           'rt_lite', 'efficientposei_lite', 'i_lite', 'efficientposeii_lite', 'ii_lite']:
		print('\n##########################################################################################################')
		print('Desired model "{0}" not available. Please select among "RT", "I", "II", "III", "IV", "RT_Lite", "I_Lite" or "II_Lite".'.format(model_name))
		print('##########################################################################################################\n')
		return False
	
	# LOAD MODEL
	else:
		model_variant = model_variant[13:] if len(model_variant) > 7 else model_variant
		lite = True if model_variant.endswith('_lite') else False
		model, resolution = get_model(framework, model_variant)
		if not model:
			return True
	
	file_path_list = sorted(glob.glob(file_path, recursive=True))  # glob
	
	for k, file_path_tmp in enumerate(file_path_list):
		print(k + 1, "*******************************************************")
		# PERFORM INFERENCE
		coordinates = analyze(video, file_path_tmp, model, framework, resolution, lite)

		# VISUALIZE PREDICTIONS
		if visualize and file_path_tmp is not None and coordinates:
			annotate(video, file_path_tmp, coordinates)
		
		# # STORE PREDICTIONS AS CSV
		# if store and coordinates:
		# 	save(video, file_path_tmp, coordinates)
	
	return True


file_path = r"E:\gldw_zt\*\*\*.jpg"
model_name = "III"
framework_name = "TFLite"
visualize = True
store = True


perform_tracking(video=False, file_path=normpath(file_path), model_name=model_name, framework_name=framework_name, visualize=visualize, store=store)

