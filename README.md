[//]: # (Image References)
[imagen1]: ./imagenes/cuadros1.JPG "Checkers before"
[imagen2]: ./imagenes/cuadrosUnd.JPG "Checkers after"
[imagen3]: ./imagenes/color1.JPG "Color before"
[imagen4]: ./imagenes/color2.JPG "Color after"
[imagen5]: ./imagenes/bird1.JPG "Bird before"
[imagen6]: ./imagenes/bird2.JPG "Bird after"
[imagen7]: ./imagenes/final.JPG "Imagen Final"
[imagen8]: ./imagenes/windows1.jpg "Polinomios"

# SDCND Project 4: Advanced Lane Finding
## Writeup


## Specification 1: Writeup, README 
### Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.

Este resumen busca cumplir con los requirision del primer punto de la especificacion.

## Specification 2: Camera Calibration
### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

El codigo se puede encontrar en el archivo DistortCorrection.py el cual incluye una clase con el mismo nombre.
La clase se llama del archivo Advanced Lines.py lineas 116 a 134
En este parte del proyecto se considera que cada camara tenga parametros de correccion de la distorsion diferentes, se crea el objeto camera en la linea 116 y se definen los parametros que son las columnas y renglones de los tableros de ajedrez, (nx, ny), en la linea 131 se llama la funcion savepick, que sirve como pipeline, dentro de la clase de llaman las funciones correspodientes como, getpoints(), se define el objeto pickle para guardar los valores calculados y no tener que estar llamando cada vez y recalculando los valores de correccion. 

## Specification 3: Pipeline (Test Images)
### Provide an example of a distortion-corrected image.
![imagen1]
![imagen2]


### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

Con la funcion def color_filter(img): en el archivo AdvancedLines.py lineas 9-23 se hace la transformacion de color de las imagenes para detectar las lineas del carril.
Se convierte de RGB a BGR, y despues a HSV, y se definen tresholds para las lineas amarillas y las lineas blancas.
Se usa para esto la funcion cv2.inRange, con los rangos para cada una de las lineas y al final se regresa una capa que muestra solo las lineas en su mayoria.
```
def color_filter(img):
    imagen_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (17, 76, 178), (30, 200, 255))

    sensitivity_1 = 68
    white = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white_2 = cv2.inRange(HSL, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    bit_layer = yellow | white | white_2 | white_3
    return bit_layer
```


Para llegar a esta solucion primero busque con las tecnicas descritas en el curso, usando Sobel, Magnitud, y Direccion, pero al probar el resultado no es tan bueno como la funcion usada, la cual es bastante robusta para el video del proyecto, y mas sencilla.
Esta tecnica esta basada en el post de Kyle Stewart-Frantz @kylesf en el canal de slack del projecto, afinando los threshols para optimizar la deteccion de las lineas amarillas.
![imagen4]


#### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Usando una imagen con el carril en linea recta stright_lines.jpg, y con el programa GNU Gimp, se identifico los puntos en la imagen original que se querian transformar, y para llevar a cabo la transformacion se uso la clase PerspectiveTransform en el archivo PerspectiveTransform.py, esta clase regresa como resultado una imagen transformada, asi como la matriz de transformacion M. 

![imagen6]

#### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Para esta parte del proyecto se siguo las instrucciones asi como parte del codigo del curso, adaptandola a la estructura del programa, para esto se uso la funcion draw_area(color_filtered, M): a partir de la linea 25 del archivo AdvancedLines.py, esta funcion toma como parametros la imagen transformada y filtrada, asi como la matriz de tranformacion M.
En la linea 70 y 71 se calcula con la formula np.polifit(leftx, lefty, 2) tomando los puntos detectados en las ventanas del arreglo  left_line_inds, tal como el ejemplo.

![imagen8]

#### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center
para la curvatura en la linea 107, se promedia los valores de la linea izqueirda y derecha, y se escala para mostrar el valor en kilometros, esto para visualizar mas facil este valor en el video.
```
curvature = ((left_curverad + right_curverad)/2)* 0.001
```
Para la desviacion del centro, se calcula la diferencia de la posicion de la linea izquierda y derecha del carril, y se escala para mostar los valores en centimetros. Esto en la linea 108
```
desviacion = (leftx_current-rightx_current)/100
```
#### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
![imagen7]


### Pipeline (video)
#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
Video can be found in YouTube, following this link.
https://youtu.be/ZEeUDFSroII

### Discussion
#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

El principal problema y lo que mas tiempo requirio fue lo relacionado al manejo de los color-spaces, debido a que las diferentes librerias usan formatos diferentes al usar matplotlib imread, se carga la imagen en RGB, pero al aplicar operaciones con OpenCV, se usa BGR, se hace complicado seguir el resultado.
En mi caso despues de afinal los paramentros en las imagenes de prueba, al procesar el video no me daba buen resultado, despues de buscar el problema, encontre que la razon es que al usar Moviepy, la imagenes a procesar entran como RGB, asi que como paso final tuve que agregar una transformacion en la linea 10, la funcion color_filter() imagen_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).

Otro problema es que hay parametros que se relacionan y funciones para calcular las lineas que regresan varios valores, en una primer instancia tuve la intension de manejar clases para las diferentes partes del proceso, pero me parecio muy complejo en la parte de la deteccion de las lineas, y preferi incluirlo como funcion en el archivo principal.

En cuanto a mejorar el resultado, en el proyecto se sugire usar una clase para seguir las lineas, y poder mantener una linea estable aun  cuando no hay una buena deteccion, para el video del proyecto no fue necesario usarla, debido a la afinacion de los parametros de la funcion color_filter(), sin embargo, para los videos de reto, challenge_video.mp4, y harder_challenge_video.mp4 se puede implementar a fin de procesarlos adecuadamente.
