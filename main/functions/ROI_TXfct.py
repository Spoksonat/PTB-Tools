import numpy as np
import pickle
import matplotlib.pyplot as plt
from bokeh import events
from bokeh.plotting import figure, show
from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar, LogTicker, BasicTicker, HoverTool, ColumnDataSource, CustomJS, Div, TextInput, TapTool, Button, Slider
from bokeh.embed import components
from bokeh.layouts import column, row
from bokeh.io import curdoc, output_notebook
from bokeh.events import Tap, SelectionGeometry
from bokeh.models.ranges import DataRange1d
from bokeh.io import curdoc
from skimage.draw import polygon

from functions.Miscellaneous import *

class ROI_TXfct:
    def __init__(self, b1maps, optsdef):
        self.optsdef    = optsdef
        self.b1maps     = b1maps
        self.masks      = None
        self.ID         = None
        self.NoOfSlices = None
        self.Slices     = None
        self.B1ID       = None
        self.src_roi    = None

    def Select_ROI_TXfct(self, doc):
        opts = self.optsdef
        B1p = self.b1maps
        B1_cxmap, type, scaling, slices, size, noofcha, rotB1, cxmap_orig = B1p.cxmap, B1p.type, B1p.scaling, B1p.slices, B1p.size, B1p.noofcha, B1p.rotB1, B1p.cxmap_orig 
        b1rsos = normalize(rms_comb(B1_cxmap))

        slider5 = Slider(start=0, end=b1rsos.shape[-1]-1, value=0, step=1, title="Image index")

        image = b1rsos[:,:,slider5.value]

        src_im     = ColumnDataSource(data={'image': [image]})
        src_im_3D  = ColumnDataSource(data={'image': [b1rsos]})
        sy,sx,_ = b1rsos.shape 
        color_mapper = LinearColorMapper(palette=opts["COLORMAP"])#LogColorMapper(palette=opts["COLORMAP"]) 
        color_mapper2 = LinearColorMapper(palette=opts["COLORMAP"])  
        src_mask = ColumnDataSource(data={'image': [np.zeros(image.shape)]})
        src_mask_3D = ColumnDataSource(data={'image': [np.zeros(b1rsos.shape)]})
        src_mask_tot = ColumnDataSource(data={'image': [np.zeros(image.shape)]})
        src_mask_tot_3D = ColumnDataSource(data={'image': [np.zeros(b1rsos.shape)]})
        src_im_masked = ColumnDataSource(data={'image': [np.zeros(image.shape)]})
        src_im_masked_3D = ColumnDataSource(data={'image': [np.zeros(b1rsos.shape)]}) 
        src_im_masked_tot = ColumnDataSource(data={'image': [np.zeros(image.shape)]})
        src_im_masked_tot_3D = ColumnDataSource(data={'image': [np.zeros(b1rsos.shape)]})
        bool_geo = ColumnDataSource(data={'bool': [0.0]})
        ROIs = [np.zeros(image.shape)] 

        source_roi  = ColumnDataSource(data={"xs": [], "ys": []})
        contour1  = ColumnDataSource(data={"xs": [], "ys": []})
        contour2  = ColumnDataSource(data={"xs": [], "ys": []})
        point1 = ColumnDataSource(data={"x": [], "y": []})
        point2 = ColumnDataSource(data={"x": [], "y": []})

        callback_tap = CustomJS(args=dict(point1=point1), code="""
                let point1_x = [cb_obj["x"]];
                let point1_y = [cb_obj["y"]];
                const new_data = {x: point1_x, y: point1_y};
                point1.data = new_data; 
                point1.change.emit();
            """)
        
        callback_tap2 = CustomJS(args=dict(point2=point2), code="""
                let point2_x = [cb_obj["x"]];
                let point2_y = [cb_obj["y"]];
                const new_data = {x: point2_x, y: point2_y};
                point2.data = new_data; 
                point2.change.emit();
            """)
        
        callback_geometry = CustomJS(args=dict(source_roi=source_roi, bool_geo=bool_geo), code="""
            let obj_geometry = cb_obj["geometry"];
            let pos_x = obj_geometry["x"];
            let pos_y = obj_geometry["y"];
            console.log(pos_x);
            let bool = {bool: [1.0]};
            bool_geo.data = bool;
            let new_data = {xs: pos_x, ys: pos_y};
            source_roi.data = new_data;
            source_roi.change.emit();
            bool_geo.change.emit();
        """)

        def callback_current_ROI():
            
            if(bool_geo.data["bool"][0] == 1.0):
                contour_x = np.array(source_roi.data["xs"]).astype(int)
                contour_y = np.array(source_roi.data["ys"]).astype(int)
                rr, cc = polygon(contour_y, contour_x, image.shape)
                mask = np.zeros(image.shape)
                mask[rr, cc] = 1.0  
                mask_3D = np.repeat(mask[:, :, np.newaxis], b1rsos.shape[-1], axis=2)
                src_mask.data = {"image": [mask]}
                src_mask_3D.data = {"image": [mask_3D]}
                bool_geo.data = {"bool": [0.0]}

            image_index = slider5.value
            src_im_masked_3D.data = {"image": [b1rsos*mask_3D]}
            src_im_masked.data = {"image": [b1rsos[:,:,image_index]*mask]}


        def callback_add_total_ROI():

            src_im_masked.data = {"image": [np.zeros(image.shape)]}

            current_mask = src_mask.data["image"][0]
            ROIs.append(current_mask)
            total_mask = src_mask_tot.data["image"][0] + current_mask
            total_mask = (total_mask >= 1).astype(int)

            current_mask_3D = src_mask_3D.data["image"][0]
            total_mask_3D = src_mask_tot_3D.data["image"][0] + current_mask_3D
            total_mask_3D = (total_mask_3D >= 1).astype(int)
            
            src_mask_tot.data = {"image": [total_mask]}
            src_mask_tot_3D.data = {"image": [total_mask_3D]}
            src_im_masked_tot_3D.data = {"image": [b1rsos*total_mask_3D]}
            image_index = slider5.value
            src_im_masked_tot.data = {"image": [src_im_masked_tot_3D.data["image"][0][:,:,image_index]]}

        def callback_substract_total_ROI():

            src_im_masked.data = {"image": [np.zeros(image.shape)]}

            current_mask = src_mask.data["image"][0]
            ROIs.append(current_mask)
            total_mask = src_mask_tot.data["image"][0]*(1 - current_mask)
            total_mask = (total_mask >= 1).astype(int)

            current_mask_3D = src_mask_3D.data["image"][0]
            total_mask_3D = src_mask_tot_3D.data["image"][0]*(1 - current_mask_3D)
            total_mask_3D = (total_mask_3D >= 1).astype(int)
            
            src_mask_tot.data = {"image": [total_mask]}
            src_mask_tot_3D.data = {"image": [total_mask_3D]}
            src_im_masked_tot_3D.data = {"image": [b1rsos*total_mask_3D]}
            image_index = slider5.value
            src_im_masked_tot.data = {"image": [src_im_masked_tot_3D.data["image"][0][:,:,image_index]]}

        def callback_intersect_total_ROI():
            
            src_im_masked.data = {"image": [np.zeros(image.shape)]}

            current_mask = src_mask.data["image"][0] 
            ROIs.append(current_mask)
            total_mask = src_mask_tot.data["image"][0] * current_mask
            total_mask = (total_mask >= 1).astype(int)

            current_mask_3D = src_mask_3D.data["image"][0]
            total_mask_3D = src_mask_tot_3D.data["image"][0]*current_mask_3D
            total_mask_3D = (total_mask_3D >= 1).astype(int)

            src_mask_tot.data = {"image": [total_mask]}
            src_mask_tot_3D.data = {"image": [total_mask_3D]}
            src_im_masked_tot_3D.data = {"image": [b1rsos*total_mask_3D]}
            image_index = slider5.value
            src_im_masked_tot.data = {"image": [src_im_masked_tot_3D.data["image"][0][:,:,image_index]]}

        def callback_change_image(attr, new, old):
           image_index = slider5.value
           src_im.data = {"image": [b1rsos[:,:,image_index]]}
           src_im_masked_tot.data = {"image": [src_im_masked_tot_3D.data["image"][0][:,:,image_index]]}
           src_im_masked.data = {"image": [src_im_masked_3D.data["image"][0][:,:,image_index]]}

        def callback_th_image(attr, new, old):
           image_index = slider5.value
           th_index = slider_th.value
           imgs = multi_otsu(src_im_masked_3D.data["image"][0])
           mask = (imgs >= 1.0).astype(int)
           src_mask_3D.data = {"image": [mask]}
           src_mask.data = {"image": [mask[:,:,image_index]]}
           src_im_masked.data = {"image": [b1rsos[:,:,image_index]*mask[:,:,image_index]]}

        def callback_save_ROI():
           np.save( "/Users/manuelfernandosanchezalarcon/Desktop/PTB_Tools/main/files/ROIs/ROI.npy" ,src_mask_tot_3D.data["image"][0])
           

        x_range = DataRange1d(start = 0, end = sx, follow="end", range_padding = 0, range_padding_units = 'percent', flipped=False)
        y_range = DataRange1d(start = 0, end = sy, follow="end", range_padding = 0, range_padding_units = 'percent', flipped=True)

        pw = 5*sx
        ph = int(pw * sy / sx)
        
        fig1 = figure(width = pw, height = ph, x_range = x_range, y_range = y_range, match_aspect=True ,tools="tap,lasso_select,wheel_zoom,zoom_in,zoom_out,reset")
        fig1.border_fill_color = "#121212"
        fig1.background_fill_color = "#121212"
        fig1.title.text_color = "white"
        fig1.xaxis.major_label_text_color = "white"
        fig1.yaxis.major_label_text_color = "white"
        fig1.image(image='image', x=0, y=0, dw=sx, dh=sy, source=src_im, color_mapper=color_mapper)
        fig1.circle(source=point1, x="x", y="y", radius=1, color="green")
        fig1.line(source=contour1, x="xs", y="ys", color="red", line_width=3)
        #color_bar1 = ColorBar(color_mapper=color_mapper, ticker=LogTicker(), label_standoff=12, location=(0,0))
        #color_bar1.background_fill_color = "#121212"
        #color_bar1.major_label_text_color = "white"
        #fig1.add_layout(color_bar1, 'right')
        #fig1.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))
        fig1.xaxis.visible = False
        fig1.yaxis.visible = False

        fig2 = figure(width = pw, height = ph, x_range = x_range, y_range = y_range, match_aspect=True ,tools="tap,lasso_select,wheel_zoom,zoom_in,zoom_out,reset")
        fig2.border_fill_color = "#121212"
        fig2.background_fill_color = "#121212"
        fig2.title.text_color = "white"
        fig2.xaxis.major_label_text_color = "white"
        fig2.yaxis.major_label_text_color = "white"
        fig2.image(image='image', x=0, y=0, dw=sx, dh=sy, source=src_im_masked, color_mapper=color_mapper2)
        color_bar2 = ColorBar(color_mapper=color_mapper2, ticker=BasicTicker(), label_standoff=12, location=(0,0))
        color_bar2.background_fill_color = "#121212"
        color_bar2.major_label_text_color = "white"
        #fig2.add_layout(color_bar2, 'right')
        #fig2.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        fig3 = figure(width = pw, height = ph, x_range = x_range, y_range = y_range, match_aspect=True ,tools="tap,lasso_select,wheel_zoom,zoom_in,zoom_out,reset")
        fig3.border_fill_color = "#121212"
        fig3.background_fill_color = "#121212"
        fig3.title.text_color = "white"
        fig3.xaxis.major_label_text_color = "white"
        fig3.yaxis.major_label_text_color = "white"
        fig3.image(image='image', x=0, y=0, dw=sx, dh=sy, source=src_im_masked_tot, color_mapper=color_mapper2)
        fig3.circle(source=point2, x="x", y="y", radius=1, color="green")
        fig3.line(source=contour2, x="xs", y="ys", color="red", line_width=3)
        color_bar3 = ColorBar(color_mapper=color_mapper2, ticker=BasicTicker(), label_standoff=12, location=(0,0))
        color_bar3.background_fill_color = "#121212"
        color_bar3.major_label_text_color = "white"
        #fig3.add_layout(color_bar3, 'right')
        #fig3.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        fig1.js_on_event(Tap, callback_tap)
        fig3.js_on_event(Tap, callback_tap2)
        fig1.js_on_event(SelectionGeometry, callback_geometry)
        fig3.js_on_event(SelectionGeometry, callback_geometry)

        button = Button(label="Add ROI", button_type="success")
        # print the source status
        button.on_click(callback_add_total_ROI)

        button2 = Button(label="Intersect ROI", button_type="success")
        # print the source status
        button2.on_click(callback_intersect_total_ROI)

        button3 = Button(label="Substract ROI", button_type="success")
        # print the source status
        button3.on_click(callback_substract_total_ROI)

        button4 = Button(label="Update ROI", button_type="success")
        # print the source status
        button4.on_click(callback_current_ROI)

        button5 = Button(label="Save ROI", button_type="success")
        # print the source status
        button5.on_click(callback_save_ROI)

        slider5.on_change("value", callback_change_image)

        slider_th = Slider(start=0, end=1, value=0, step=1, title="Th-index")
        slider_th.on_change("value", callback_th_image)
        

        text_region_growing = Div(text="Region growing tol:")
        text_contour_threshold = Div(text="Contour threshold:")
        
        layout = row(column(fig1, fig3), column(fig2, row(button, button2, button3, button4, button5), row(slider5), row(slider_th)))

        doc.add_root(layout)
        