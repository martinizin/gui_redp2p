
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
            }
        ]);

        const enlaces = new vis.DataSet([
            { from: 1, to: 3, label: '' }
        ]);

        const container = document.getElementById('red');
        const opciones = {
            physics: { enabled: false },
            interaction: { tooltipDelay: 10, hover: true }
        };

        const red = new vis.Network(container, { nodes: nodos, edges: enlaces }, opciones);

        let selectedNode = null;
        let tooltip;

        // Tooltip functionality
        red.on("hoverNode", function(event) {
            const nodeId = event.node;
            const nodeData = nodos.get(nodeId);
            tooltip = document.createElement('tool-tip');

            // Display node parameters in the tooltip
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

        // Click functionality to open modal
        red.on("click", function(event) {
            const nodeId = event.nodes[0];
            if (nodeId) {
                selectedNode = nodos.get(nodeId);

                // Populate the modal with node data
                document.getElementById('nodeLabel').value = selectedNode.label;
                document.getElementById('nodePotencia').value = selectedNode.potencia;

                // Populate additional parameters
                const additionalParamsDiv = document.getElementById('additionalParams');
                additionalParamsDiv.innerHTML = '';
                for (const key in selectedNode.parametros) {
                    additionalParamsDiv.innerHTML += `
                        <div class="mb-3">
                            <label for="${key}" class="form-label">${key}</label>
                            <input type="text" class="form-control" id="${key}" name="${key}" value="${selectedNode.parametros[key]}">
                        </div>
                    `;
                }

                // Show the modal
                const modal = new bootstrap.Modal(document.getElementById('editNodeModal'));
                modal.show();
            }
        });

        // Save changes from modal
        document.getElementById('saveNodeChanges').addEventListener('click', function() {
            if (selectedNode) {
                // Update node data
                selectedNode.label = document.getElementById('nodeLabel').value;
                selectedNode.potencia = parseFloat(document.getElementById('nodePotencia').value);

                // Update additional parameters
                const additionalParamsDiv = document.getElementById('additionalParams');
                const inputs = additionalParamsDiv.querySelectorAll('input');
                inputs.forEach(input => {
                    selectedNode.parametros[input.name] = input.value;
                });

                // Update the node in the dataset
                nodos.update(selectedNode);

                // Close the modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('editNodeModal'));
                modal.hide();

                // Optionally, update results or perform recalculations here
                alert('Cambios guardados. Actualiza los resultados según sea necesario.');
            }
        });

        // Handle fiber span selection
        document.getElementById('fiberSpanSelector').addEventListener('change', function(event) {
            const numSpans = parseInt(event.target.value);

            // Clear existing nodes and edges
            nodos.clear();
            enlaces.clear();

            // Add emitter node
            nodos.add({
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
            });

            // Add fiber spans and intermediate nodes
            let previousNodeId = 1;
            for (let i = 1; i <= numSpans; i++) {
                const fiberNodeId = 100 + i; // Unique ID for fiber spans
                const intermediateNodeId = 200 + i; // Unique ID for intermediate nodes

                // Prompt user for fiber span length
                const fiberLength = prompt(`Ingrese la longitud del Tramo de Fibra ${i} (en km):`, "23");

                // Add fiber span node
                nodos.add({
                    id: fiberNodeId,
                    label: `Tramo de Fibra ${i}`,
                    x: 100 + i * 150,
                    y: 100,
                    potencia: 24.85,
                    loss_coef: 0.2,
                    att_in: 0,
                    con_in: 0.25,
                    con_out: 0.30,
                    longitud: fiberLength,
                    shape: 'ellipse',
                    parametros: {
                        'Coeficiente de Pérdida': '0.2 dB/km',
                        'Atenuación en la Entrada': '0 dB',
                        'Conector de Entrada': '0.25 dB',
                        'Conector de Salida': '0.30 dB',
                        'Longitud': `${fiberLength} km`
                    }
                });

                // Add intermediate node (no label)
                nodos.add({
                    id: intermediateNodeId,
                    label: '',
                    x: 100 + i * 150 + 50,
                    y: 100,
                    shape: 'circle',
                    parametros: {}
                });

                // Add edges
                enlaces.add({ from: previousNodeId, to: fiberNodeId, label: '' });
                enlaces.add({ from: fiberNodeId, to: intermediateNodeId, label: '' });

                previousNodeId = intermediateNodeId;
            }

            // Add receiver node
            nodos.add({
                id: 3,
                label: 'Receptor',
                x: 100 + (numSpans + 1) * 150,
                y: 100,
                potencia: 17.90,
                sensibilidad_receptor_dbm: 14,
                shape: 'box',
                parametros: {
                    'Sensibilidad del Receptor': '14.00 dBm'
                }
            });

            // Connect the last intermediate node to the receiver
            enlaces.add({ from: previousNodeId, to: 3, label: '' });
        });
    }

    dibujarRed();
