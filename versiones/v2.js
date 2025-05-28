
        // Dibujar la red inicial con un solo tramo de fibra
        function dibujarRed() {
            const nodos = new vis.DataSet([
                { 
                    id: 1, 
                    label: 'Emisor', 
                    x: 100, 
                    y: 100, 
                    potencia: 30, 
                    unidad_potencia: 'dBm', 
                    shape: 'box',
                    parametros: {
                        'Unidad de Potencia': 'dBm',
                        'Potencia de Entrada': '30.00 dBm'
                    }
                },
                { 
                    id: 3, 
                    label: 'Receptor', 
                    x: 500, 
                    y: 100, 
                    potencia: 17.90, 
                    sensibilidad_receptor_dbm: 14, 
                    shape: 'box',
                    parametros: {
                        'Sensibilidad del Receptor': '14.00 dBm'
                    }
                },
                { 
                    id: 2, 
                    label: 'Tramo de Fibra',
                    x: 300, 
                    y: 100, 
                    potencia: 24.85, 
                    loss_coef: 0.2, 
                    att_in: 0, 
                    con_in: 0.25, 
                    con_out: 0.30, 
                    longitud: 23,
                    parametros: {
                        'Coeficiente de Pérdida': '0.2 dB/km',
                        'Atenuación en la Entrada': '0 dB',
                        'Conector de Entrada': '0.25 dB',
                        'Conector de Salida': '0.30 dB'
                    },
                
                }
            ]);

            const enlaces = new vis.DataSet([
                { from: 1, to: 3, label: '', longitud: 23 },  // Conectar directamente el Emisor con la Fibra
                   // Conectar directamente la Fibra con el Receptor
            ]);

            const container = document.getElementById('red');
            const opciones = {
                physics: { enabled: false },
                interaction: { tooltipDelay: 10, hover: true }
            };

            const red = new vis.Network(container, { nodes: nodos, edges: enlaces }, opciones);

            let tooltip;

            red.on("hoverNode", function(event) {
                const nodeId = event.node;
                const nodeData = nodos.get(nodeId);
                tooltip = document.createElement('tool-tip');

                // Mostrar los parámetros del nodo en el tooltip
                let tooltipContent = `<strong>${nodeData.label}</strong><br>`;
                for (const key in nodeData.parametros) {
                    tooltipContent += `<strong>${key}:</strong> ${nodeData.parametros[key]}<br>`;
                }
                tooltip.innerHTML = tooltipContent;

                document.body.appendChild(tooltip);

                const position = event.event.pointer;
                tooltip.style.left = position.x + 'px';
                tooltip.style.top = position.y + 'px';

                tooltip.style.opacity = 1;
            });

            red.on("blurNode", function() {
                if (tooltip) {
                    tooltip.remove();
                    tooltip = null;
                }
            });

            red.on("click", function(event) {
                const nodeId = event.nodes[0]; // Obtener ID del nodo clicado
                const nodeData = nodos.get(nodeId);

                // Aquí podrías abrir un formulario para editar los valores
                alert(`Haz clic en el nodo: ${nodeData.label}`);
            });
        }

        dibujarRed();
