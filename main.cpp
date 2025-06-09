#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

//  FUNCIONES DE ACTIVACIÓN

// Función Sigmoide: convierte cualquier número a rango 0-1
double sigmoide(double valor) {
    return 1.0 / (1.0 + exp(-valor));
}

// Derivada de la sigmoide para retropropagación
double derivada_sigmoide(double valor) {
    double sig = sigmoide(valor);
    return sig * (1 - sig);
}

// Función ReLU
double relu(double valor) {
    return max(0.0, valor);
}

// Derivada de ReLU
double derivada_relu(double valor) {
    return valor > 0 ? 1.0 : 0.0;
}

// números a probabilidades que suman 1
vector<double> softmax(const vector<double>& valores) {
    vector<double> probabilidades(valores.size());
    double suma_total = 0.0;
    double valor_maximo = *max_element(valores.begin(), valores.end());

    // Calcular exponenciales
    for (size_t i = 0; i < valores.size(); ++i) {
        probabilidades[i] = exp(valores[i] - valor_maximo);
        suma_total += probabilidades[i];
    }

    // Normalizar para que sumen 1
    for (size_t i = 0; i < valores.size(); ++i) {
        probabilidades[i] /= suma_total;
    }

    return probabilidades;
}

//  OPERACIONES BÁSICAS

// Multiplica pesos por entradas y suma todo
double producto_punto(const vector<double>& pesos, const vector<double>& entradas) {
    double resultado = 0.0;
    for (size_t i = 0; i < pesos.size(); ++i) {
        resultado += pesos[i] * entradas[i];
    }
    return resultado;
}

//  RED NEURONAL MNIST
class RedNeuronalMNIST {
private:

    double velocidad_aprendizaje;
    int num_pixeles;        // 784 píxeles
    int num_neuronas_ocultas;   // 128 neuronas
    int num_digitos;        // 10 dígitos

    // Matrices de pesos
    vector<vector<double>> pesos_entrada_a_oculta;  // De píxeles a capa oculta
    vector<vector<double>> pesos_oculta_a_salida;   // De capa oculta a dígitos

public:
    // Constructor: inicializa la red
    RedNeuronalMNIST(int pixeles = 784, int ocultas = 128, int digitos = 10, double velocidad = 0.1)
        : num_pixeles(pixeles), num_neuronas_ocultas(ocultas), num_digitos(digitos), velocidad_aprendizaje(velocidad) {

        srand(time(0));
        cout << "\n Inicializando Red Neuronal...\n";
        cout << "    Píxeles de entrada: " << pixeles << "\n";
        cout << "    Neuronas ocultas: " << ocultas << "\n";
        cout << "    Dígitos de salida: " << digitos << "\n";

        // Inicializar pesos con valores pequeños aleatorios
        double limite_entrada = sqrt(6.0 / (pixeles + ocultas));
        double limite_salida = sqrt(6.0 / (ocultas + digitos));

        // matriz de pesos entrada a oculta
        pesos_entrada_a_oculta = vector<vector<double>>(ocultas, vector<double>(pixeles + 1));
        for (int neurona = 0; neurona < ocultas; ++neurona) {
            for (int conexion = 0; conexion < pixeles + 1; ++conexion) {
                pesos_entrada_a_oculta[neurona][conexion] =
                    ((double)rand() / RAND_MAX) * 2 * limite_entrada - limite_entrada;
            }
        }

        // Crear matriz de pesos oculta a salida
        pesos_oculta_a_salida = vector<vector<double>>(digitos, vector<double>(ocultas + 1));
        for (int digito = 0; digito < digitos; ++digito) {
            for (int conexion = 0; conexion < ocultas + 1; ++conexion) {
                pesos_oculta_a_salida[digito][conexion] =
                    ((double)rand() / RAND_MAX) * 2 * limite_salida - limite_salida;
            }
        }
    }

    // número (0-9) a [0,0,0,1,0,0,0,0,0,0]
    vector<double> crear_objetivo(int digito_correcto) {
        vector<double> objetivo(num_digitos, 0.0);
        objetivo[digito_correcto] = 1.0;
        return objetivo;
    }

    // Normaliza
    vector<double> normalizar_imagen(const vector<double>& pixeles_crudos) {
        vector<double> pixeles_normalizados(pixeles_crudos.size());
        for (size_t i = 0; i < pixeles_crudos.size(); ++i) {
            pixeles_normalizados[i] = pixeles_crudos[i] / 255.0;
        }
        return pixeles_normalizados;
    }

    // ENTRENAMIENTO
    void entrenar(const vector<vector<double>>& imagenes, const vector<int>& digitos_correctos, int num_epocas) {
        cout << "\n Iniciando entrenamiento con " << imagenes.size() << " imágenes...\n";
        cout << "================================================\n";

        for (int epoca = 0; epoca < num_epocas; ++epoca) {
            double error_total = 0.0;
            int predicciones_correctas = 0;

            // Procesar cada imagen
            for (size_t img = 0; img < imagenes.size(); ++img) {
                // Preparar datos
                vector<double> pixeles_norm = normalizar_imagen(imagenes[img]);
                pixeles_norm.insert(pixeles_norm.begin(), 1.0); // Agregar sesgo
                vector<double> objetivo = crear_objetivo(digitos_correctos[img]);

                //  PROPAGACIÓN HACIA ADELANTE

                // Capa oculta: calcular activaciones
                vector<double> valores_ocultos(num_neuronas_ocultas);
                vector<double> activaciones_ocultas(num_neuronas_ocultas);

                for (int neurona = 0; neurona < num_neuronas_ocultas; ++neurona) {
                    valores_ocultos[neurona] = producto_punto(pesos_entrada_a_oculta[neurona], pixeles_norm);
                    activaciones_ocultas[neurona] = relu(valores_ocultos[neurona]);
                }
                activaciones_ocultas.insert(activaciones_ocultas.begin(), 1.0); // Sesgo

                // Capa de salida: calcular probabilidades
                vector<double> valores_salida(num_digitos);
                for (int digito = 0; digito < num_digitos; ++digito) {
                    valores_salida[digito] = producto_punto(pesos_oculta_a_salida[digito], activaciones_ocultas);
                }

                vector<double> probabilidades = softmax(valores_salida);

                // Calcular error (cross-entropy)
                for (int i = 0; i < num_digitos; ++i) {
                    error_total -= objetivo[i] * log(max(probabilidades[i], 1e-15));
                }

                // Verificar si la predicción es correcta
                int prediccion = max_element(probabilidades.begin(), probabilidades.end()) - probabilidades.begin();
                if (prediccion == digitos_correctos[img]) {
                    predicciones_correctas++;
                }

                //  RETROPROPAGACIÓN

                // Calcular errores de la capa de salida
                vector<double> errores_salida(num_digitos);
                for (int i = 0; i < num_digitos; ++i) {
                    errores_salida[i] = probabilidades[i] - objetivo[i];
                }

                // Calcular errores de la capa oculta
                vector<double> errores_ocultos(num_neuronas_ocultas);
                for (int neurona = 0; neurona < num_neuronas_ocultas; ++neurona) {
                    double suma_error = 0.0;
                    for (int digito = 0; digito < num_digitos; ++digito) {
                        suma_error += errores_salida[digito] * pesos_oculta_a_salida[digito][neurona + 1];
                    }
                    errores_ocultos[neurona] = suma_error * derivada_relu(valores_ocultos[neurona]);
                }

                //  pesos oculta salida
                for (int digito = 0; digito < num_digitos; ++digito) {
                    for (int conexion = 0; conexion < num_neuronas_ocultas + 1; ++conexion) {
                        pesos_oculta_a_salida[digito][conexion] -=
                            velocidad_aprendizaje * errores_salida[digito] * activaciones_ocultas[conexion];
                    }
                }

                // de pesos entrada oculta
                for (int neurona = 0; neurona < num_neuronas_ocultas; ++neurona) {
                    for (int conexion = 0; conexion < num_pixeles + 1; ++conexion) {
                        pesos_entrada_a_oculta[neurona][conexion] -=
                            velocidad_aprendizaje * errores_ocultos[neurona] * pixeles_norm[conexion];
                    }
                }
            }

            // Mostrar progreso cada 10 épocas
            if ((epoca + 1) % 10 == 0) {
                double precision = (double)predicciones_correctas / imagenes.size() * 100.0;
                cout << " Época " << epoca + 1 << "/" << num_epocas
                     << " | Error: " << (error_total/imagenes.size())
                     << " | Precisión: " << precision << "%\n";
            }
        }
        cout << "================================================\n";
        cout << " Entrenamiento completado!\n";
    }

    // PREDICCIÓN
    int predecir_digito(const vector<double>& imagen_cruda) {
        // Normalizar y agregar sesgo
        vector<double> pixeles_norm = normalizar_imagen(imagen_cruda);
        pixeles_norm.insert(pixeles_norm.begin(), 1.0);

        // Capa oculta
        vector<double> activaciones_ocultas(num_neuronas_ocultas);
        for (int neurona = 0; neurona < num_neuronas_ocultas; ++neurona) {
            double valor = producto_punto(pesos_entrada_a_oculta[neurona], pixeles_norm);
            activaciones_ocultas[neurona] = relu(valor);
        }
        activaciones_ocultas.insert(activaciones_ocultas.begin(), 1.0);

        // Capa de salida
        vector<double> valores_salida(num_digitos);
        for (int digito = 0; digito < num_digitos; ++digito) {
            valores_salida[digito] = producto_punto(pesos_oculta_a_salida[digito], activaciones_ocultas);
        }

        vector<double> probabilidades = softmax(valores_salida);

        // Retornar dígito con mayor probabilidad
        return max_element(probabilidades.begin(), probabilidades.end()) - probabilidades.begin();
    }

    // Obtener todas las probabilidades
    vector<double> obtener_probabilidades(const vector<double>& imagen_cruda) {
        vector<double> pixeles_norm = normalizar_imagen(imagen_cruda);
        pixeles_norm.insert(pixeles_norm.begin(), 1.0);

        vector<double> activaciones_ocultas(num_neuronas_ocultas);
        for (int neurona = 0; neurona < num_neuronas_ocultas; ++neurona) {
            double valor = producto_punto(pesos_entrada_a_oculta[neurona], pixeles_norm);
            activaciones_ocultas[neurona] = relu(valor);
        }
        activaciones_ocultas.insert(activaciones_ocultas.begin(), 1.0);

        vector<double> valores_salida(num_digitos);
        for (int digito = 0; digito < num_digitos; ++digito) {
            valores_salida[digito] = producto_punto(pesos_oculta_a_salida[digito], activaciones_ocultas);
        }

        return softmax(valores_salida);
    }
};

// Leer CSV de MNIST
bool leer_csv_mnist(const string& mnist_train.csv, vector<vector<double>>& imagenes, vector<int>& etiquetas) {
    ifstream archivo(mnist_train.csv);
    if (!archivo.is_open()) {
        cout << "Error: No se pudo abrir el archivo " << mnist_train.csv << endl;
        return false;
    }

    string linea;
    bool primera_linea = true;
    int contador = 0;

    cout << "Leyendo archivo: " << mnist_train.csv << "...\n";

    while (getline(archivo, linea)) {
        // Saltar la primera línea si contiene headers
        if (primera_linea) {
            primera_linea = false;
            // Verificar si la primera línea contiene headers (no números)
            if (linea.find("label") != string::npos || linea.find("pixel") != string::npos) {
                continue;
            }
        }

        stringstream ss(linea);
        string celda;
        vector<double> fila_datos;

        // Leer cada valor separado por comas
        while (getline(ss, celda, ',')) {
            try {
                double valor = stod(celda);
                fila_datos.push_back(valor);
            } catch (const exception& e) {
                cout << "Error al convertir: " << celda << endl;
                continue;
            }
        }

        // Verificar que la fila tenga el tamaño correcto
        if (fila_datos.size() == 785) { // 784 píxeles + 1 etiqueta
            // Primera columna es la etiqueta
            int etiqueta = (int)fila_datos[0];
            etiquetas.push_back(etiqueta);

            // Las siguientes 784 columnas son los píxeles
            vector<double> imagen(fila_datos.begin() + 1, fila_datos.end());
            imagenes.push_back(imagen);

            contador++;

            // Mostrar progreso cada 10000 imágenes
            if (contador % 10000 == 0) {
                cout << "Procesadas " << contador << " imágenes...\n";
            }
        } else if (fila_datos.size() == 784) {
            // Caso alternativo: solo píxeles (sin etiqueta)
            vector<double> imagen(fila_datos.begin(), fila_datos.end());
            imagenes.push_back(imagen);
            etiquetas.push_back(0); // Etiqueta por defecto
            contador++;
        }
    }

    archivo.close();
    cout << "Lectura completada. Total de imágenes: " << contador << "\n";
    return contador > 0;
}

// NUEVA FUNCIÓN: Evaluar modelo con datos de prueba
void evaluar_modelo(RedNeuronalMNIST& red, const vector<vector<double>>& imagenes_test,
                   const vector<int>& etiquetas_test) {
    cout << "\n  EVALUACIÓN CON DATOS DE PRUEBA \n";
    cout << "========================================\n";

    int predicciones_correctas = 0;
    int total_muestras = min(1000, (int)imagenes_test.size()); // Evaluar máximo 1000 muestras

    for (int i = 0; i < total_muestras; ++i) {
        int prediccion = red.predecir_digito(imagenes_test[i]);
        if (prediccion == etiquetas_test[i]) {
            predicciones_correctas++;
        }

        // Mostrar progreso
        if ((i + 1) % 100 == 0) {
            cout << "Evaluadas " << (i + 1) << "/" << total_muestras << " muestras\n";
        }
    }

    double precision = (double)predicciones_correctas / total_muestras * 100.0;
    cout << "\n📊 RESULTADOS DE EVALUACIÓN:\n";
    cout << "   Muestras evaluadas: " << total_muestras << "\n";
    cout << "   Predicciones correctas: " << predicciones_correctas << "\n";
    cout << "   Precisión: " << precision << "%\n";
}

//  GENERACIÓN DE DATOS DE PRUEBA (FALLBACK)
void generar_datos_prueba(vector<vector<double>>& imagenes, vector<int>& etiquetas) {
    cout << "\n Generando datos de prueba simulados...\n";

    srand(42); // Semilla fija para reproducibilidad

    for (int i = 0; i < 1000; ++i) {
        vector<double> imagen(784);
        int digito_real = rand() % 10; // Dígito aleatorio 0-9

        // Generar patrón único para cada dígito
        for (int pixel = 0; pixel < 784; ++pixel) {
            // Crear patrón basado en el dígito real
            double intensidad = (sin(pixel * 0.01 + digito_real) + 1) * 127.5;
            intensidad += ((double)rand() / RAND_MAX - 0.5) * 50; // Añadir ruido
            imagen[pixel] = max(0.0, min(255.0, intensidad));
        }

        imagenes.push_back(imagen);
        etiquetas.push_back(digito_real);
    }

    cout << " Se generaron " << imagenes.size() << " imágenes de prueba\n";
}

//  FUNCIÓN PRINCIPAL MODIFICADA
int main() {
    cout << "  Red de Clasificación de Dígitos MNIST \n";
    cout << "================================================================\n";

    // Configuración de la red
    const int PIXELES_ENTRADA = 784;    // 28x28 píxeles
    const int NEURONAS_OCULTAS = 128;   // Capa intermedia
    const int DIGITOS_SALIDA = 10;      // Números 0-9
    const double VELOCIDAD_APRENDIZAJE = 0.1;

    // Crear la red neuronal
    RedNeuronalMNIST mi_red(PIXELES_ENTRADA, NEURONAS_OCULTAS, DIGITOS_SALIDA, VELOCIDAD_APRENDIZAJE);

    // Intentar leer datos de entrenamiento desde CSV
    vector<vector<double>> imagenes_entrenamiento;
    vector<int> etiquetas_entrenamiento;

    cout << "\n CARGANDO DATOS DE ENTRENAMIENTO\n";
    bool csv_cargado = leer_csv_mnist("mnist_train.csv", imagenes_entrenamiento, etiquetas_entrenamiento);

    if (!csv_cargado) {
        cout << "⚠  No se encontró mnist_train.csv, usando datos simulados...\n";
        generar_datos_prueba(imagenes_entrenamiento, etiquetas_entrenamiento);
    } else {
        // Opcional: Usar solo una muestra para entrenamiento rápido
        int muestras_entrenamiento = min(5000, (int)imagenes_entrenamiento.size());
        imagenes_entrenamiento.resize(muestras_entrenamiento);
        etiquetas_entrenamiento.resize(muestras_entrenamiento);

        cout << "Usando " << muestras_entrenamiento << " muestras para entrenamiento\n";
    }

    // Entrenar la red
    mi_red.entrenar(imagenes_entrenamiento, etiquetas_entrenamiento, csv_cargado ? 50 : 100);

    // Intentar leer datos de prueba
    vector<vector<double>> imagenes_prueba;
    vector<int> etiquetas_prueba;

    cout << "\n CARGANDO DATOS DE PRUEBA\n";
    bool test_csv_cargado = leer_csv_mnist("mnist_test.csv", imagenes_prueba, etiquetas_prueba);

    if (test_csv_cargado) {
        evaluar_modelo(mi_red, imagenes_prueba, etiquetas_prueba);
    } else {
        cout << "️  No se encontró mnist_test.csv, usando datos de entrenamiento para demostración...\n";
        imagenes_prueba = imagenes_entrenamiento;
        etiquetas_prueba = etiquetas_entrenamiento;
    }

    //  DEMOSTRACIÓN
    cout << "\n MUESTRAS DE DEMOSTRACIÓN \n";
    cout << "================================\n";

    for (int prueba = 0; prueba < 5 && prueba < imagenes_prueba.size(); ++prueba) {
        int prediccion = mi_red.predecir_digito(imagenes_prueba[prueba]);
        vector<double> todas_probabilidades = mi_red.obtener_probabilidades(imagenes_prueba[prueba]);

        cout << "\n Muestra " << prueba + 1 << ":\n";
        cout << "   Dígito correcto: " << etiquetas_prueba[prueba] << "\n";
        cout << "   Predicción: " << prediccion << "\n";
        cout << "   Confianza: " << (int)(todas_probabilidades[prediccion] * 100) << "%\n";
        cout << "   Probabilidades por dígito:\n      ";

        for (int digito = 0; digito < 10; ++digito) {
            cout << digito << ":" << (int)(todas_probabilidades[digito] * 100) << "% ";
        }
        cout << "\n";

        // Indicar si la predicción fue correcta
        if (prediccion == etiquetas_prueba[prueba]) {
            cout << "    Predicción CORRECTA!\n";
        } else {
            cout << "    Predicción incorrecta\n";
        }
    }

    cout << "\nPROCESO COMPLETADO \n";
    cout << "La red neuronal ha sido entrenada y evaluada exitosamente!\n";

    return 0;
}
