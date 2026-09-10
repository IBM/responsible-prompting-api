function renderGraph(graphData, svgId, tooltipId) {
    const width = 600;
    const height = 300;
    const marginTop = 30;
    const marginRight = 30;
    const marginBottom = 30;
    const marginLeft = 30;
    const nodeRadius = 3;

    const svg = d3
    .select(svgId)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("style", "max-width: 100%; height: auto; font: 8px sans-serif;");

    const tooltip = d3.select(tooltipId);

    let { nodes, edges } = graphData;

    if (nodes.length === 0) {
        svg.selectAll("*").remove();
        svg
            .append("text")
            .attr("x", width / 2)
            .attr("y", height / 2)
            .attr("text-anchor", "middle")
            .attr("fill", "#666")
            .text("No nodes to display");
        return;
    }

    const xDomain = d3.extent(nodes, (d) => d.x);
    const yDomain = d3.extent(nodes, (d) => d.y);
    const xPadding = 2;
    const yPadding = 2;

    const xScale = d3
        .scaleLinear()
        .domain([xDomain[0] - xPadding, xDomain[1] + xPadding])
        .nice()
        .range([marginLeft, width - marginRight]);

    const yScale = d3
        .scaleLinear()
        .domain([yDomain[0] - yPadding, yDomain[1] + yPadding])
        .nice()
        .range([height - marginBottom, marginTop]);

    svg.selectAll("*").remove();

    // X axis
    svg
        .append("g")
        .attr("transform", `translate(0, ${height - marginBottom})`)
        .call(d3.axisBottom(xScale).ticks(width / 80))
        .call((g) => g.select(".domain").remove())
        .call((g) =>
            g
                .append("text")
                .attr("x", width)
                .attr("y", marginBottom - 4)
                .attr("fill", "currentColor")
                .attr("text-anchor", "end")
                .text("Semantic dimension 1")
        );

    // Y axis
    svg
        .append("g")
        .attr("transform", `translate(${marginLeft}, 0)`)
        .call(d3.axisLeft(yScale))
        .call((g) => g.select(".domain").remove())
        .call((g) =>
            g
                .append("text")
                .attr("x", -marginLeft)
                .attr("y", 10)
                .attr("fill", "currentColor")
                .attr("text-anchor", "start")
                .text("Semantic dimension 2")
        );

    // Grid
    svg
        .append("g")
        .attr("stroke", "#cccccc")
        .attr("stroke-opacity", 0.5)
        .call((g) =>
            g
                .append("g")
                .selectAll("line")
                .data(xScale.ticks())
                .join("line")
                .attr("x1", (d) => 0.5 + xScale(d))
                .attr("x2", (d) => 0.5 + xScale(d))
                .attr("y1", marginTop)
                .attr("y2", height - marginBottom)
        )
        .call((g) =>
            g
                .append("g")
                .selectAll("line")
                .data(yScale.ticks())
                .join("line")
                .attr("y1", (d) => 0.5 + yScale(d))
                .attr("y2", (d) => 0.5 + yScale(d))
                .attr("x1", marginLeft)
                .attr("x2", width - marginRight)
        );

    // Edges
    svg
        .append("g")
        .selectAll("line")
        .data(edges)
        .join("line")
        .attr("stroke", "#666")
        .attr("stroke-opacity", 0.5)
        .attr(
            "x1",
            (d) =>
                xScale(d.source.x) +
                (d.source.x < d.target.x ? 1.3 * nodeRadius : -1.3 * nodeRadius)
        )
        .attr("y1", (d) => yScale(d.source.y))
        .attr(
            "x2",
            (d) =>
                xScale(d.target.x) +
                (d.source.x > d.target.x ? 1.3 * nodeRadius : -1.3 * nodeRadius)
        )
        .attr("y2", (d) => yScale(d.target.y))
        .style("stroke-dasharray", (d) =>
            d.target.type === "add" ? "3,3" : ""
        );

    // Nodes
    svg
        .append("g")
        .attr("stroke-width", 2.5)
        .attr("stroke-opacity", 0.5)
        .attr("fill", "none")
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("stroke", (d) =>
            d.type === "add" ? "green" : d.type === "input" ? "#666" : "red"
        )
        .attr("cx", (d) => xScale(d.x))
        .attr("cy", (d) => yScale(d.y))
        .attr("r", nodeRadius);

    // Labels
    svg
        .append("g")
        .attr("font-family", "sans-serif")
        .attr("text-opacity", 0.5)
        .attr("font-size", 8)
        .selectAll("text")
        .data(nodes)
        .join("text")
        .attr("dy", "0.35em")
        .attr("x", (d) => xScale(d.x) + 5)
        .attr("y", (d) => yScale(d.y))
        .text((d) => d.label)
        .on("mousemove", function (event, d) {
            d3.select(this)
                .transition()
                .duration(50)
                .attr("text-opacity", 1.0)
                .attr("stroke", "white")
                .attr("stroke-width", 3)
                .style("paint-order", "stroke fill")
                .attr(
                    "fill",
                    d.type === "add"
                        ? "green"
                        : d.type === "input"
                            ? "black"
                            : "red"
                );
            tooltip.transition().duration(50).style("opacity", 1);
            tooltip
                .html(`<strong>${d.label}:</strong><br/>${d.text}`)
                .style("left", event.pageX + 10 + "px")
                .style("top", event.pageY + 10 + "px");
        })
        .on("mouseout", function () {
            d3.select(this)
                .transition()
                .duration(50)
                .attr("text-opacity", 0.5)
                .attr("stroke-width", 0)
                .style("paint-order", "fill")
                .attr("fill", "black");
            tooltip.transition().duration(50).style("opacity", 0);
        });
}

function generateGraph(recommendations) {
    const rec = recommendations;
    let i = 0,
        j = 0;

    const graphData = { nodes: [], edges: [] };

    // Input sentences
    if (rec.input && rec.input.length > 0) {
        graphData.nodes.push({
            id: 0,
            x: Number(rec.input[0].x),
            y: Number(rec.input[0].y),
            text: rec.input[0].sentence,
            label: "S1",
            type: "input",
        });
        for (i = 1; i < rec.input.length; i++) {
            graphData.nodes.push({
                id: i,
                x: Number(rec.input[i].x),
                y: Number(rec.input[i].y),
                text: rec.input[i].sentence,
                label: `S${i + 1}`,
                type: "input",
            });
            graphData.edges.push({ source: i - 1, target: i, type: "input" });
        }
    }

    // “Add” recommendations
    if (rec.add && rec.add.length > 0) {
        for (j = 0; j < rec.add.length; j++) {
            graphData.nodes.push({
                id: i + j,
                x: Number(rec.add[j].x),
                y: Number(rec.add[j].y),
                text: rec.add[j].prompt,
                label: rec.add[j].value,
                type: "add",
            });
            graphData.edges.push({
                source: i - 1,
                target: i + j,
                type: "add",
            });
        }
    }

    // “Remove” recommendation (first only)
    if (rec.remove && rec.remove.length > 0) {
        graphData.nodes.push({
            id: i + j,
            x: Number(rec.remove[0].x),
            y: Number(rec.remove[0].y),
            text: rec.remove[0].closest_harmful_sentence,
            label: rec.remove[0].value,
            type: "remove",
        });
    }

    graphData.edges = graphData.edges.map((e) => ({
        source: graphData.nodes.find((n) => n.id === e.source),
        target: graphData.nodes.find((n) => n.id === e.target),
        type: e.type,
    }));

    return graphData;
}

function appendUserTurn(turn, chatId) {
    const bubble = $("<div>")
        .addClass("message user")
        .html(marked.parse(turn.content));
    $(chatId).append(bubble);

    if (turn.recs && turn.recs.length > 0) {
        const container = $("<div>").addClass("recs-container");
        turn.recs.forEach((rec, index) => {
            let r = rec.recommendation;
            const item = $("<div>")
                .addClass("recs-item")
                .addClass(r.type === "add" ? "add" : "remove")
                .attr('title', r.sentence)
                .text((r.type === "add" ? "+ " : "x ") + r.value);
            let itemId = `rec-${conversation.length}-${index}`;
            item.attr('id', itemId);
            container.append(item);
        });
        container.css("align-self", "flex-end");
        $(chatId).append(container);
    }
    $(chatId).scrollTop($(chatId)[0].scrollHeight);
}

function appendAssistantTurn(text, chatId, bubbleId) {
    const bubble = $("<div>")
        .addClass("message assistant")
        .attr('id', bubbleId)
        .text(text);
    $(chatId).append(bubble);
    $(chatId).scrollTop($(chatId)[0].scrollHeight);
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function generateResponse(rawText, chatId, sendBtnId) {
    const userText = rawText.trim();
    if (!userText) return;

    generating = true;
    const thisTurn = {
        role: "user",
        content: userText,
        recs: selectedRecs.slice(),
    };
    conversation.push(thisTurn);
    appendUserTurn(thisTurn, chatId);

    selectedRecs = [];

    // "Typing..." placeholder
    const typingBubble = $("<div>")
        .addClass("message assistant")
        .attr("id", "typing")
        .text("Requesting Content…");
    $(chatId).append(typingBubble);
    $(chatId).scrollTop($(chatId)[0].scrollHeight);

    // Build full prompt
    let fullPrompt = "";
    conversation.forEach((turn) => {
        if (turn.role === "user") {
            fullPrompt += `User: ${turn.content}\n`;
        } else {
            fullPrompt += `Assistant: ${turn.content}\n`;
        }
    });
    fullPrompt += "Assistant: ";

    modelId = selectedModel.id;
    inference_provider = selectedModel.inference_provider;

    $.ajax({
        url: "/demo_inference?prompt=" + encodeURIComponent(fullPrompt) + "&model_id=" + encodeURIComponent(modelId) + "&inference_provider=" + encodeURIComponent(inference_provider),
        dataType: "json",
        success: async function (data) {
            $("#typing").remove();
            const generated = data.content.trim();
            const modelId = data.model_id;
            const temp = data.temperature;
            const maxTokens = data.max_new_tokens;

            const modelHeader = $("<div style='display: flex; flex-direction: row;'>")
                .addClass("model-info")
                .html(`<img src="./imgs/granite.svg" class='icon'/> <div style='display:flex; align-items: center;margin-left: 0.5rem'>${modelId} (via ${inference_provider})</div>`);
            $(chatId).append(modelHeader);

            appendAssistantTurn("", chatId, "assistantBubble")

            const chars = generated.split("");
            let idx = 0;
            const chatEl = $(chatId)[0];
            var keepScrollDown = true;
            var cur = '';
            while (idx < chars.length) {
                if (chatEl.scrollHeight != chatEl.scrollTop + chatEl.clientHeight) {
                    keepScrollDown = false;
                }
                cur += chars[idx];
                $("#assistantBubble").html(marked.parse(cur));
                idx++;
                if(keepScrollDown) $(chatId).scrollTop($(chatId)[0].scrollHeight);
                
                await delay(0);
            } 
            if(keepScrollDown) $(chatId).scrollTop($(chatId)[0].scrollHeight);
            $("#assistantBubble").removeAttr("id");
            conversation.push({
                role: "assistant",
                content: generated,
            });
            generating = false;
            if($("#userInputDiv").text().length > 0) $(sendBtnId).attr("disabled", false);
        },
        error: function (xhr) {
            $("#typing").remove();
            const err =
                xhr.responseJSON?.error?.message ||
                "Unknown error from inference.";
            appendAssistantTurn(`Error: ${err}`, chatId, "assistantBubble");
            generating = false;
        },
    });
}

let popupOpen = false;
function toggleGraph(data, itemId, chatId) {
    const existing = document.getElementById('popup');
    if (existing) {
        existing.remove();
        popupOpen = false;
        return;
    }
    const tag = document.getElementById(itemId);
    const rect = tag.getBoundingClientRect();
    const popup = document.createElement('div');
    popup.id = 'popup';
    // reuse .tooltip styles but allow pointer events
    popup.className = 'tooltip';
    popup.style.pointerEvents = 'auto';
    popup.style.width = '800px';
    popup.style.height = '420px';
    popup.style.zIndex = 2;
    popup.style.left = `${rect.left + window.scrollX - 17}px`;
    popup.style.top  = `${rect.top + window.scrollY - rect.height - 410}px`;

    const container = document.createElement('svg');
    container.style.width = '100%';
    container.style.height = '100%';

    container.innerHTML = "<svg id='popup-graph'></svg>"
    popup.appendChild(container);
    document.body.appendChild(popup);

    renderGraph(data, '#popup-graph', "#tooltip");
    // close on outside click
    const onClickOutside = (e) => {
        if (!popup.contains(e.target) && e.target !== tag) {
            popup.remove();
            document.removeEventListener('click', onClickOutside);
            popupOpen = false;
        }
    };
    setTimeout(() => {
        document.addEventListener('click', onClickOutside);
    }, 0);
    popupOpen = true;
}

let lastRecommendations = [];
const conversation = []; // stores { role, content, recs? }
let selectedRecs = []; // stores { recommendation, graphData }
let debounceId = null;
function generateRecommendations(sendBtnId, promptInputId, recommendationDivId) {
    const txt = $(promptInputId).text().trim() || "";

    clearTimeout(debounceId);

    let stopRecos = false;

    if (txt.length > 0 && /[.?!]$/.test(txt)) {
        debounceId = setTimeout(() => {
            $(recommendationDivId).html(`
                <div class="bx--tag rec-tags-inputarea rec-tags-inputarea-loading">
                </div>
            `);
            $.getJSON("/recommend?prompt=" + encodeURIComponent(txt), (data) => {
                if (stopRecos) return;
                $(recommendationDivId).empty();
                lastRecommendations = data;
                graphData = generateGraph(lastRecommendations);

                if (
                    (!data.add || data.add.length === 0) &&
                    (!data.remove || data.remove.length === 0)
                ) {
                    $(recommendationDivId).text("No recommendations found.");
                } else {
                    const graphBtn = $(`
                        <button id="graphBtn" class="btn" title="View sentences in the embedding space">
                            <img src="./imgs/data-vis.svg" alt="Graph" class="icon" height="2rem"/>
                        </button>
                    `);
                    graphBtn.on('click', () => toggleGraph(graphData, "graphBtn", 'afas'));
                    $(recommendationDivId).append(graphBtn);
                }

                if (data.remove && data.remove.length > 0) {
                    const rec = data.remove[0];
                    const sentence = rec.sentence.replaceAll("'", "\\'");
                    const valueEscaped = rec.value.replaceAll("'", "\\'");
                    const $tag = $(`
                        <div class="bx--tag bx--tag--red bx--tag--deletable rec-tags-inputarea">
                        ✕ ${valueEscaped}
                        </div>
                    `);
                    // Removal recommendations
                    $tag.hover(
                        () => {
                            // Snapshot the current content as cloned DOM nodes so it can be
                            // restored on hover-out without round-tripping through an HTML string.
                            $(promptInputId).data("prevNodes", $(promptInputId).contents().clone(true));
                            $(promptInputId).html(
                                $(promptInputId).html().replace(
                                    rec.sentence.trim(),
                                    " <span class='rec-span' style='text-decoration: line-through; background-color: lightcoral;'>" + rec.sentence.trim() + "</span>"
                                )
                            )
                        },
                        () => {
                            const prevNodes = $(promptInputId).data("prevNodes");
                            if (prevNodes && prevNodes.length) {
                                $(promptInputId).empty().append(prevNodes.clone(true));
                            } else {
                                $(promptInputId).text(txt);
                            }
                        }
                    );

                    $tag.click(() => {
                        const updated = $(promptInputId)
                            .text()
                            .replace(rec.sentence, "")
                            .replace(/ {2,}/g, " ")
                            .trim();
                        $(promptInputId).text(updated);
                        selectedRecs.push({
                            'recommendation': { type: "remove", value: rec.value, sentence: rec.sentence },
                            'graphData': graphData,
                        });
                        $(recommendationDivId).empty();
                        $(promptInputId).trigger("input");
                    });

                    $(recommendationDivId).append($tag);
                }

                if (data.add && data.add.length > 0) {
                    data.add.forEach((rec) => {
                        if (!$(promptInputId).text().includes(rec.prompt)) {
                            const promptEscaped = rec.prompt.replaceAll("'", "\\'");
                            const valueEscaped = rec.value.replaceAll("'", "\\'");
                            const $tag = $(`
                                <div class="bx--tag bx--tag--green rec-tags-inputarea">
                                + ${valueEscaped}
                                </div>
                            `);
                            
                            // Addition recommendations
                            $tag.hover(
                                () => {
                                    const cur = $(promptInputId).html();
                                    // prevHtml is still consumed as a string by the click handler below;
                                    // prevNodes is the cloned-node snapshot used for a safe hover-out restore.
                                    $(promptInputId).data("prevHtml", cur);
                                    $(promptInputId).data("prevNodes", $(promptInputId).contents().clone(true));
                                    $(promptInputId).html($(promptInputId).html() + " <span class='rec-span' style='background-color: green;'>" + rec.prompt.trim() + "</span>")
                                    $(promptInputId).scrollTop($(promptInputId)[0].scrollHeight);
                                },
                                () => {
                                    const prevNodes = $(promptInputId).data("prevNodes");
                                    if (prevNodes && prevNodes.length) {
                                        $(promptInputId).empty().append(prevNodes.clone(true));
                                    } else {
                                        $(promptInputId).text(txt);
                                    }
                                }
                            );

                            $tag.click(() => {
                                const base =
                                    $(promptInputId).data("prevHtml") ||
                                    $(promptInputId).html().trim();
                                $(promptInputId).html((base + " " + rec.prompt).trim());
                                selectedRecs.push({
                                    'recommendation': { type: "add", value: rec.value, sentence: rec.prompt },
                                    'graphData': graphData,
                                });
                                $(recommendationDivId).empty();
                                $(promptInputId).trigger("input");
                            });

                            $(recommendationDivId).append($tag);
                        }
                    });
                }
            });
        }, 500);
        $(sendBtnId).on("click", function () {
            stopRecos = true;
        })
    } else {
        $(recommendationDivId).empty();
        lastRecommendations = null;
    }
}