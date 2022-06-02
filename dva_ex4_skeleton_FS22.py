"""
How to run the script:

This script does not need a bokeh server, simnply run it with
´´´
python dva_ex4_skeleton_FS22.py
´´´


Point Distribution:

1 Point: Divergence Plot
1 Point: Vorticity Plot
1.5 Points: Vector Coloring
1.5 Points: Hedgehog Overlays

"""

import numpy as np
import os
import bokeh

from bokeh.layouts import layout, row
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, ColumnDataSource
from colorcet import CET_L16

from colorsys import hsv_to_rgb


output_file('DVA_ex4.html')
color = CET_L16

HEDGEHOG_OPACITY = 0.85
HEDGEHOG_GRID_SIZE = 10
SLICE = 20

def to_bokeh_image(rgba_uint8):
    " Essentially converts an rgba image of uint8 type to a image usable by bokeh "
    if len(rgba_uint8.shape) > 2 \
            and int(bokeh.__version__.split(".")[0]) >= 2 \
            and int(bokeh.__version__.split(".")[1]) >= 2:

        np_img2d = np.zeros((rgba_uint8.shape[0], rgba_uint8.shape[1]), dtype=np.uint32)
        view = np_img2d.view(dtype=np.uint8).reshape(rgba_uint8.shape)
        view[:] = rgba_uint8[:]
    else:
        np_img2d = rgba_uint8
    return [np_img2d]


def get_divergence(vx_wind, vy_wind):
    # Use np.gradient to calculate the gradient of a vector field. Find out what exactly the return values represent and
    # use the appropriate elements for your calculations
    vx_wind = vx_wind[:,:,SLICE]
    vy_wind = vy_wind[:,:,SLICE]
    arr = [vx_wind, vy_wind]

    div_v = np.ufunc.reduce(np.add, [np.gradient(arr[i], axis=i) for i in range(2)])
    # your code

    return div_v


def get_vorticity(vx_wind, vy_wind):
    # Calculate the gradient again and use the appropriate results to calculate the vorticity. Think about what happens
    # to the z-component and the derivatives with respect to z for a two dimensional vector field.
    # (You can save the gradient in the divergence calculations or recalculate it here. Since the gradient function is
    # fast and we have rather small data slices the impact of recalculating it is negligible.)
    vx_wind = vx_wind[:,:,SLICE]
    vy_wind = vy_wind[:,:,SLICE]

    arr = [np.gradient(vx_wind, axis=1), np.gradient(vy_wind, axis=0)]

    vort_v = np.ufunc.reduce(np.subtract, arr)
    return vort_v


# calculates the HSV colors of the xy-windspeed vectors and maps them to RGBA colors
def vector_color_coding(vx_wind, vy_wind):
    vx_wind = vx_wind[:, :, SLICE]
    vy_wind = vy_wind[:, :, SLICE]

    # The brightness value (V) is set on the normalized magnitude of the vector
    norm = np.linalg.norm([vx_wind.flatten(), vy_wind.flatten()], axis=0)
    maximum = np.max(norm)
    v = norm/maximum
    # Calculate the hue (H) as the angle between the vector and the positive x-axis
    # your code
    angles = np.arctan2(vx_wind, vy_wind)
    ang = np.array([angle if angle > 0 else 2*np.pi + angle for angle in angles.flatten()])
    hue = ang / (2*np.pi)

    # Saturation (S) can be set to 1
    S = 1

    # Either use colorsys.hsv_to_rgb or implement the color conversion yourself using the
    # algorithm for the HSV to RGB conversion, see https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    rgba_colors = np.reshape([hsv_to_rgb(h, S, v_q) + (1,) for h,v_q in zip(hue,v)], (vx_wind.shape[0], vx_wind.shape[1],4))*255
    # your code
    # The RGBA colors have to be saved as uint8 for the bokeh plot to properly work
    return rgba_colors.astype('uint8')


def get_hedgehog(vx_wind, vy_wind):
    # Compute start (x0, y0) and end coordinates (x1, y1) for the hedgehog plot
    # and return a ColumnDataSource
    vx_wind = vx_wind[:,:,SLICE]
    vy_wind = vy_wind[:,:,SLICE]

    norm_x = np.linalg.norm([vx_wind, vy_wind], axis=0)
    norm_y = np.linalg.norm([vx_wind, vy_wind], axis=0)



    # To reduce the density, only pick every HEDGEHOG_GRID_SIZEth coordinate of the vector field
    # so if HEDGEHOG_GRID_SIZE is 10 then only pick every 10th grid line, you can do this with the numpy indexing
    # https://stackoverflow.com/questions/25876640/subsampling-every-nth-entry-in-a-numpy-array

    # Have a look at the numpy.indices function https://numpy.org/doc/stable/reference/generated/numpy.indices.html
    # To reduce the cluttered visuals, scale the length of the hedgehog such that they are not longer
    # than one grid cell diagonal. Have a look at the np.linalg.norm function
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    return ColumnDataSource(dict(
        x0=norm_x,
        y0=norm_y,
        x1=norm_x,
        y1=norm_y,
    ))


# load and process the required data
print('processing data')
x_wind_file = 'Uf24.bin'
x_wind_path = os.path.abspath(os.path.dirname(x_wind_file))
x_wind_data = np.fromfile(os.path.join(x_wind_path, x_wind_file), dtype=np.dtype('>f'))
x_wind_data = np.reshape(x_wind_data, [500, 500, 100], order='F')
x_wind_data = np.flipud(x_wind_data)

# replace the missing "no data" values with the average of the dataset
filtered_average = np.average(x_wind_data[x_wind_data < 1e35])
x_wind_data[x_wind_data == 1e35] = filtered_average

y_wind_file = 'Vf24.bin'
y_wind_path = os.path.abspath(os.path.dirname(y_wind_file))
y_wind_data = np.fromfile(os.path.join(y_wind_path, y_wind_file), dtype=np.dtype('>f'))
y_wind_data = np.reshape(y_wind_data, [500, 500, 100], order='F')
y_wind_data = np.flipud(y_wind_data)

# replace the missing "no data" values with the average of the dataset
filtered_average = np.average(y_wind_data[y_wind_data < 1e35])
y_wind_data[y_wind_data == 1e35] = filtered_average

wind_vcc = vector_color_coding(x_wind_data, y_wind_data)
wind_divergence = get_divergence(x_wind_data, y_wind_data)
wind_vorticity = get_vorticity(x_wind_data, y_wind_data)
print('data processing completed')

# Compute the Hedgehog ColumnDataSource
source = get_hedgehog(x_wind_data, y_wind_data)

fig_args = {'x_range': (0, 500), 'y_range': (0, 500), 'width': 500, 'height': 400, 'toolbar_location': None,
            'active_scroll': 'wheel_zoom'}
img_args = {'dh': 500, 'dw': 500, 'x': 0, 'y': 0}
cb_args = {'ticker': BasicTicker(), 'label_standoff': 12, 'border_line_color': None, 'location': (0, 0)}

# divergence plot
color_mapper_divergence = LinearColorMapper(palette=CET_L16, low=np.amin(wind_divergence),
                                            high=np.amax(wind_divergence))
divergence_plot = figure(title="Divergence", **fig_args)
divergence_plot.image(image=to_bokeh_image(wind_divergence), color_mapper=color_mapper_divergence, **img_args)
divergence_color_bar = ColorBar(color_mapper=color_mapper_divergence, **cb_args)
divergence_plot.add_layout(divergence_color_bar, 'right')

color_mapper_vorticity = LinearColorMapper(palette=CET_L16, low=np.amin(wind_vorticity), high=np.amax(wind_vorticity))
plot_vorticity=figure(title="Vorticity", **fig_args)
plot_vorticity.image(image=to_bokeh_image(wind_vorticity), color_mapper=color_mapper_vorticity, **img_args)
color_bar_vorticity = ColorBar(color_mapper=color_mapper_vorticity, **cb_args)
plot_vorticity.add_layout(color_bar_vorticity, "right")
plot_vorticity.segment(color="white", line_alpha=HEDGEHOG_OPACITY, source=source)

color_plot=figure(title="Vector Color Coding", **fig_args)
color_plot.image_rgba(image=to_bokeh_image(wind_vcc), **img_args)
color_plot.segment(color="white", line_alpha=HEDGEHOG_OPACITY, source=source)

lt=row(divergence_plot, plot_vorticity, color_plot, sizing_mode="scale_width")
show(lt)
# TODO use figure.segment to overlay a hedgehog plot over the divergence plot
# TODO Set the opacity with HEDGEHOG_OPACITY

# TODO create vorticity plot in a similar fashion as the divergence plot and overlay a hedgehog plot


# TODO create vector color coding plot and overlay a hedgehog plot


# TODO create and show plot layout
