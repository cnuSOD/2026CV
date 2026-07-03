/*
前端交互脚本
功能包括：
1. 上传图片本地预览
2. 滑块数值实时显示
3. 模型信息联动更新
4. 错误弹窗显示与关闭
5. 检测结果查看器：SVG 框层、悬停提示、点击详情
6. 放大查看器：滚轮缩放、拖拽平移、点击命中检测框
*/

document.addEventListener("DOMContentLoaded", function () {
    const SVG_NS = "http://www.w3.org/2000/svg";
    const MODEL_PROFILES = window.MODEL_PROFILES || {};
    const ERROR_MESSAGE = String(window.ERROR_MESSAGE || "").trim();
    const RESULT_DETECTIONS = Array.isArray(window.RESULT_DETECTIONS) ? window.RESULT_DETECTIONS : [];

    const imageInput = document.getElementById("image");
    const previewImage = document.getElementById("uploadPreview");
    const confRange = document.getElementById("conf_threshold");
    const iouRange = document.getElementById("iou_threshold");
    const confValue = document.getElementById("confValue");
    const iouValue = document.getElementById("iouValue");
    const modelSelect = document.getElementById("model_name");
    const resetBtn = document.getElementById("resetBtn");

    const modelTitle = document.getElementById("modelTitle");
    const modelParams = document.getElementById("modelParams");
    const modelDesc = document.getElementById("modelDesc");

    const errorModalBackdrop = document.getElementById("errorModalBackdrop");
    const errorModalMessage = document.getElementById("errorModalMessage");
    const errorModalClose = document.getElementById("errorModalClose");
    const errorModalConfirm = document.getElementById("errorModalConfirm");

    const openViewerModalBtn = document.getElementById("openViewerModalBtn");
    const viewerModalBackdrop = document.getElementById("viewerModalBackdrop");
    const viewerModalClose = document.getElementById("viewerModalClose");
    const viewerResetBtn = document.getElementById("viewerResetBtn");

    const detectionRows = Array.from(document.querySelectorAll(".detection-row"));
    const detectionMap = new Map(
        RESULT_DETECTIONS.map(function (item) {
            return [String(item.index), item];
        })
    );

    const colorPalette = [
        { stroke: "#0f766e", fill: "rgba(15, 118, 110, 0.10)" },
        { stroke: "#2563eb", fill: "rgba(37, 99, 235, 0.10)" },
        { stroke: "#0891b2", fill: "rgba(8, 145, 178, 0.10)" },
        { stroke: "#ea580c", fill: "rgba(234, 88, 12, 0.10)" },
        { stroke: "#dc2626", fill: "rgba(220, 38, 38, 0.10)" },
        { stroke: "#7c3aed", fill: "rgba(124, 58, 237, 0.10)" }
    ];

    function pickColor(detection) {
        const classId = Number(detection.class_id || 0);
        const index = ((classId % colorPalette.length) + colorPalette.length) % colorPalette.length;
        return colorPalette[index];
    }

    function getDetection(index) {
        return detectionMap.get(String(index)) || null;
    }

    function updateRangeText() {
        if (confRange && confValue) {
            confValue.textContent = Number(confRange.value).toFixed(2);
        }
        if (iouRange && iouValue) {
            iouValue.textContent = Number(iouRange.value).toFixed(2);
        }
    }

    function updateModelInfo() {
        if (!modelSelect) {
            return;
        }

        const selected = modelSelect.value;
        const profile = MODEL_PROFILES[selected] || {};

        if (modelTitle) {
            modelTitle.textContent = profile.title || "YOLO 模型";
        }
        if (modelParams) {
            modelParams.textContent = profile.params || "待补充";
        }
        if (modelDesc) {
            modelDesc.textContent = profile.description || "适合课程展示使用。";
        }
    }

    function openErrorModal(message) {
        if (!errorModalBackdrop || !errorModalMessage || !message) {
            return;
        }
        errorModalMessage.textContent = message;
        errorModalBackdrop.hidden = false;
    }

    function closeErrorModal() {
        if (errorModalBackdrop) {
            errorModalBackdrop.hidden = true;
        }
    }

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function buildInfoCardHtml(detection) {
        return [
            "<div class=\"viewer-info-head\">",
            "<span class=\"viewer-info-index\">#" + escapeHtml(detection.index) + "</span>",
            "<span class=\"viewer-info-class\">" + escapeHtml(detection.class_name) + "</span>",
            "</div>",
            "<div class=\"viewer-info-grid\">",
            "<div><span>置信度</span><strong>" + Number(detection.confidence).toFixed(4) + "</strong></div>",
            "<div><span>是否为小目标</span><strong>" + (detection.is_small_target ? "是" : "否") + "</strong></div>",
            "<div><span>宽度</span><strong>" + escapeHtml(detection.width) + " px</strong></div>",
            "<div><span>高度</span><strong>" + escapeHtml(detection.height) + " px</strong></div>",
            "</div>",
            "<div class=\"viewer-info-bbox\">",
            "<span>bbox 坐标</span>",
            "<code>[" + [detection.x1, detection.y1, detection.x2, detection.y2].join(", ") + "]</code>",
            "</div>"
        ].join("");
    }

    function createViewer(config) {
        const stage = document.getElementById(config.stageId);
        const canvas = document.getElementById(config.canvasId);
        const image = document.getElementById(config.imageId);
        const overlay = document.getElementById(config.overlayId);
        const tooltip = document.getElementById(config.tooltipId);
        const infoCard = document.getElementById(config.infoCardId);
        const modeGroup = config.modeGroupId ? document.getElementById(config.modeGroupId) : null;
        const emptyNode = config.emptyId ? document.getElementById(config.emptyId) : null;

        if (!stage || !canvas || !image || !overlay || !tooltip || !infoCard) {
            return null;
        }

        const state = {
            mode: config.defaultMode || "boxes",
            hoveredId: null,
            activeId: null,
            scale: 1,
            minScale: 1,
            maxScale: 6,
            translateX: 0,
            translateY: 0,
            isDragging: false,
            dragMoved: false,
            dragStartX: 0,
            dragStartY: 0,
            pointerDownX: 0,
            pointerDownY: 0,
            rafId: 0
        };

        const imageWidth = Number(stage.dataset.imageWidth || 0);
        const imageHeight = Number(stage.dataset.imageHeight || 0);
        overlay.setAttribute("viewBox", "0 0 " + imageWidth + " " + imageHeight);

        function syncModeButtons() {
            if (!modeGroup) {
                return;
            }
            Array.from(modeGroup.querySelectorAll(".mode-btn")).forEach(function (button) {
                const isActive = button.dataset.mode === state.mode;
                button.classList.toggle("is-active", isActive);
                button.setAttribute("aria-pressed", String(isActive));
            });
        }

        function applyCanvasTransform() {
            canvas.style.transform = "translate(" + state.translateX + "px, " + state.translateY + "px) scale(" + state.scale + ")";
        }

        function clampTranslation() {
            if (!config.enablePanZoom) {
                state.translateX = 0;
                state.translateY = 0;
                return;
            }

            const rect = stage.getBoundingClientRect();
            const contentWidth = rect.width * state.scale;
            const contentHeight = rect.height * state.scale;
            const maxOffsetX = Math.max(0, (contentWidth - rect.width) / 2);
            const maxOffsetY = Math.max(0, (contentHeight - rect.height) / 2);

            state.translateX = Math.min(maxOffsetX, Math.max(-maxOffsetX, state.translateX));
            state.translateY = Math.min(maxOffsetY, Math.max(-maxOffsetY, state.translateY));
        }

        function scheduleRender() {
            if (state.rafId) {
                return;
            }
            state.rafId = window.requestAnimationFrame(function () {
                state.rafId = 0;
                renderOverlay();
            });
        }

        function clearTooltip() {
            tooltip.hidden = true;
            tooltip.textContent = "";
        }

        function clearInfoCard() {
            infoCard.hidden = true;
            infoCard.innerHTML = "";
        }

        function updateTooltipPosition(clientX, clientY) {
            const stageRect = stage.getBoundingClientRect();
            tooltip.style.left = clientX - stageRect.left + 16 + "px";
            tooltip.style.top = clientY - stageRect.top + 16 + "px";
        }

        function showTooltip(detection, clientX, clientY) {
            tooltip.textContent = config.enablePanZoom
                ? "#" + detection.index + " " + detection.class_name + "，点击序号查看详情"
                : "#" + detection.index + " " + detection.class_name;
            tooltip.hidden = false;
            updateTooltipPosition(clientX, clientY);
        }

        function imageToStagePoint(x, y) {
            const rect = stage.getBoundingClientRect();
            const scaleX = rect.width / imageWidth;
            const scaleY = rect.height / imageHeight;

            return {
                x: state.translateX + (x * scaleX - rect.width / 2) * state.scale + rect.width / 2,
                y: state.translateY + (y * scaleY - rect.height / 2) * state.scale + rect.height / 2
            };
        }

        function showInfoCard(detection) {
            const stageRect = stage.getBoundingClientRect();

            infoCard.innerHTML = buildInfoCardHtml(detection);
            infoCard.hidden = false;

            const cardWidth = infoCard.offsetWidth || 280;
            const cardHeight = infoCard.offsetHeight || 200;
            const gap = 12;
            const labelAnchor = overlay.querySelector("[data-label-anchor=\"" + String(detection.index) + "\"]");

            if (!labelAnchor) {
                const fallbackPoint = imageToStagePoint(detection.x1, Math.max(6, detection.y1 - 28));
                infoCard.style.left = Math.max(12, Math.min(stageRect.width - cardWidth - 12, fallbackPoint.x - cardWidth - gap)) + "px";
                infoCard.style.top = Math.max(12, Math.min(stageRect.height - cardHeight - 12, fallbackPoint.y - cardHeight / 2)) + "px";
                return;
            }

            const labelRect = labelAnchor.getBoundingClientRect();
            const labelLeft = labelRect.left - stageRect.left;
            const labelRight = labelRect.right - stageRect.left;
            const labelCenterY = labelRect.top - stageRect.top + labelRect.height / 2;

            let left = labelLeft - cardWidth - gap;
            let top = labelCenterY - cardHeight / 2;

            if (left < 12) {
                left = labelRight + gap;
            }
            if (left + cardWidth > stageRect.width - 12) {
                left = Math.max(12, stageRect.width - cardWidth - 12);
            }
            if (top < 12) {
                top = 12;
            }
            if (top + cardHeight > stageRect.height - 12) {
                top = Math.max(12, stageRect.height - cardHeight - 12);
            }

            infoCard.style.left = left + "px";
            infoCard.style.top = top + "px";
        }

        function stageToImagePoint(clientX, clientY) {
            /*
            坐标映射逻辑：
            1. 鼠标先得到在当前 stage 里的相对位置
            2. 再减去当前平移量 translateX / translateY
            3. 再除以当前缩放 scale
            4. 最后映射回原图坐标，后续命中检测框都基于原图坐标做判断
            */
            const rect = stage.getBoundingClientRect();
            const localX = clientX - rect.left;
            const localY = clientY - rect.top;

            return {
                x: (((localX - rect.width / 2) - state.translateX) / state.scale + rect.width / 2) * (imageWidth / rect.width),
                y: (((localY - rect.height / 2) - state.translateY) / state.scale + rect.height / 2) * (imageHeight / rect.height)
            };
        }

        function hitTestDetection(clientX, clientY) {
            /*
            点击命中检测框逻辑：
            1. 把鼠标点位换回原图坐标
            2. 遍历所有 bbox，找出包含该点的目标
            3. 命中多个框时优先返回面积更小的框，便于密集小目标场景操作
            */
            const point = stageToImagePoint(clientX, clientY);
            const hits = RESULT_DETECTIONS.filter(function (item) {
                return point.x >= item.x1 && point.x <= item.x2 && point.y >= item.y1 && point.y <= item.y2;
            });

            hits.sort(function (a, b) {
                if (a.area !== b.area) {
                    return a.area - b.area;
                }
                return a.index - b.index;
            });

            return hits[0] || null;
        }

        function buildLabelText(detection) {
            if (state.mode === "boxes") {
                return "";
            }
            if (state.mode === "index") {
                return "#" + detection.index;
            }
            if (state.mode === "full") {
                return "#" + detection.index + " " + detection.class_name + " " + Number(detection.confidence).toFixed(2);
            }
            return "";
        }

        function renderOverlay() {
            overlay.innerHTML = "";
            overlay.setAttribute("viewBox", "0 0 " + imageWidth + " " + imageHeight);

            RESULT_DETECTIONS.forEach(function (detection) {
                const color = pickColor(detection);
                const isHovered = String(detection.index) === state.hoveredId;
                const isActive = String(detection.index) === state.activeId;
                const group = document.createElementNS(SVG_NS, "g");
                const rect = document.createElementNS(SVG_NS, "rect");

                rect.setAttribute("x", detection.x1);
                rect.setAttribute("y", detection.y1);
                rect.setAttribute("width", Math.max(1, detection.x2 - detection.x1));
                rect.setAttribute("height", Math.max(1, detection.y2 - detection.y1));
                rect.setAttribute("rx", 6);
                rect.setAttribute("ry", 6);
                rect.setAttribute("fill", isActive ? color.fill : "rgba(255,255,255,0.01)");
                rect.setAttribute("stroke", isActive || isHovered ? "#0f766e" : color.stroke);
                rect.setAttribute("stroke-width", isActive ? 3.5 : (isHovered ? 3 : 2));
                group.appendChild(rect);

                const labelText = buildLabelText(detection);
                if (labelText) {
                    const labelWidth = Math.max(44, labelText.length * 7.5 + 16);
                    const labelX = detection.x1;
                    const labelY = Math.max(6, detection.y1 - 28);

                    const labelBg = document.createElementNS(SVG_NS, "rect");
                    labelBg.setAttribute("x", labelX);
                    labelBg.setAttribute("y", labelY);
                    labelBg.setAttribute("width", labelWidth);
                    labelBg.setAttribute("height", 22);
                    labelBg.setAttribute("rx", 11);
                    labelBg.setAttribute("ry", 11);
                    labelBg.setAttribute("fill", isActive ? "#0f766e" : "rgba(255,255,255,0.92)");
                    labelBg.setAttribute("stroke", isActive ? "#0f766e" : "rgba(15, 118, 110, 0.20)");
                    labelBg.setAttribute("data-label-anchor", String(detection.index));
                    group.appendChild(labelBg);

                    const text = document.createElementNS(SVG_NS, "text");
                    text.setAttribute("x", labelX + 8);
                    text.setAttribute("y", labelY + 15);
                    text.setAttribute("class", "viewer-svg-label");
                    text.setAttribute("fill", isActive ? "#ffffff" : "#183b56");
                    text.setAttribute("data-label-anchor", String(detection.index));
                    text.textContent = labelText;
                    group.appendChild(text);

                    if (config.enablePanZoom) {
                        const labelHit = document.createElementNS(SVG_NS, "rect");
                        labelHit.setAttribute("x", labelX);
                        labelHit.setAttribute("y", labelY);
                        labelHit.setAttribute("width", labelWidth);
                        labelHit.setAttribute("height", 22);
                        labelHit.setAttribute("rx", 11);
                        labelHit.setAttribute("ry", 11);
                        labelHit.setAttribute("fill", "rgba(255,255,255,0.001)");
                        labelHit.setAttribute("data-label-hit", String(detection.index));
                        labelHit.setAttribute("data-label-anchor", String(detection.index));
                        group.appendChild(labelHit);
                    }
                }

                overlay.appendChild(group);
            });

            if (state.activeId && config.enablePanZoom) {
                const detection = getDetection(state.activeId);
                if (detection) {
                    showInfoCard(detection);
                }
            } else {
                clearInfoCard();
            }
        }

        function updateMode(mode) {
            state.mode = mode;
            syncModeButtons();
            clearTooltip();
            if (!config.enablePanZoom) {
                clearInfoCard();
                state.activeId = null;
                updateGlobalSelection(null, config.name);
            }

            renderOverlay();
        }

        function hitTestLabel(clientX, clientY) {
            const element = document.elementFromPoint(clientX, clientY);
            if (!element || !overlay.contains(element)) {
                return null;
            }

            const hitId = element.getAttribute("data-label-hit");
            return hitId ? getDetection(hitId) : null;
        }

        function resetView() {
            state.scale = 1;
            state.translateX = 0;
            state.translateY = 0;
            applyCanvasTransform();
            renderOverlay();
        }

        function open() {
            if (emptyNode) {
                emptyNode.hidden = RESULT_DETECTIONS.length > 0;
            }
            resetView();
        }

        function close() {
            state.isDragging = false;
            clearTooltip();
        }

        function onWheel(event) {
            if (!config.enablePanZoom || !RESULT_DETECTIONS.length) {
                return;
            }

            event.preventDefault();

            /*
            缩放和平移逻辑：
            1. 滚轮调整 scale
            2. 以鼠标位置为缩放中心同步修正平移量
            3. 这样放大时能稳定锁定局部区域，而不是每次都从中心跳动
            */
            const rect = stage.getBoundingClientRect();
            const localX = event.clientX - rect.left;
            const localY = event.clientY - rect.top;
            const offsetX = localX - rect.width / 2 - state.translateX;
            const offsetY = localY - rect.height / 2 - state.translateY;
            const zoomFactor = event.deltaY < 0 ? 1.12 : 0.9;
            const nextScale = Math.min(state.maxScale, Math.max(state.minScale, state.scale * zoomFactor));

            if (nextScale === state.scale) {
                return;
            }

            state.translateX -= offsetX * (nextScale / state.scale - 1);
            state.translateY -= offsetY * (nextScale / state.scale - 1);
            state.scale = nextScale;
            clampTranslation();
            applyCanvasTransform();
            scheduleRender();
        }

        function onPointerDown(event) {
            if (!config.enablePanZoom || event.button !== 0) {
                return;
            }
            state.isDragging = true;
            state.dragMoved = false;
            state.dragStartX = event.clientX - state.translateX;
            state.dragStartY = event.clientY - state.translateY;
            state.pointerDownX = event.clientX;
            state.pointerDownY = event.clientY;
            stage.classList.add("is-dragging");
        }

        function onPointerMove(event) {
            const labelHit = config.enablePanZoom ? hitTestLabel(event.clientX, event.clientY) : null;
            const hit = labelHit || hitTestDetection(event.clientX, event.clientY);

            if (state.isDragging) {
                if (Math.abs(event.clientX - state.pointerDownX) > 3 || Math.abs(event.clientY - state.pointerDownY) > 3) {
                    state.dragMoved = true;
                }
                state.translateX = event.clientX - state.dragStartX;
                state.translateY = event.clientY - state.dragStartY;
                clampTranslation();
                applyCanvasTransform();
                scheduleRender();
                clearTooltip();
                return;
            }

            if (hit) {
                state.hoveredId = String(hit.index);
                if (state.mode !== "focus" || !state.activeId) {
                    /*
                    tooltip 显示逻辑：
                    悬停时只显示轻量提示，避免所有标签同时展开。
                    */
                    showTooltip(hit, event.clientX, event.clientY);
                }
            } else {
                state.hoveredId = null;
                clearTooltip();
            }

            renderOverlay();
        }

        function onPointerUp() {
            state.isDragging = false;
            stage.classList.remove("is-dragging");
        }

        function onClick(event) {
            if (state.dragMoved) {
                state.dragMoved = false;
                return;
            }

            const labelHit = config.enablePanZoom ? hitTestLabel(event.clientX, event.clientY) : null;
            const hit = config.enablePanZoom ? labelHit : hitTestDetection(event.clientX, event.clientY);
            if (!hit) {
                if (config.enablePanZoom) {
                    state.activeId = null;
                    updateGlobalSelection(null, config.name);
                    clearInfoCard();
                    renderOverlay();
                }
                return;
            }

            if (state.activeId === String(hit.index)) {
                state.activeId = null;
                updateGlobalSelection(null, config.name);
                clearInfoCard();
                renderOverlay();
                return;
            }

            state.activeId = String(hit.index);
            updateGlobalSelection(state.activeId, config.name);

            renderOverlay();
        }

        function mount() {
            syncModeButtons();
            applyCanvasTransform();
            renderOverlay();

            if (modeGroup) {
                Array.from(modeGroup.querySelectorAll(".mode-btn")).forEach(function (button) {
                    button.addEventListener("click", function () {
                        updateMode(button.dataset.mode);
                    });
                });
            }

            image.addEventListener("load", renderOverlay);
            stage.addEventListener("click", onClick);
            stage.addEventListener("mousemove", onPointerMove);
            stage.addEventListener("mouseleave", function () {
                state.hoveredId = null;
                clearTooltip();
                renderOverlay();
            });

            if (config.enablePanZoom) {
                stage.addEventListener("wheel", onWheel, { passive: false });
                stage.addEventListener("mousedown", onPointerDown);
                window.addEventListener("mousemove", onPointerMove);
                window.addEventListener("mouseup", onPointerUp);
            }

            window.addEventListener("resize", scheduleRender);
        }

        return {
            name: config.name,
            mount: mount,
            open: open,
            close: close,
            resetView: resetView,
            scrollIntoView: function () {
                stage.scrollIntoView({ behavior: "smooth", block: "nearest" });
            },
            setExternalActive: function (index) {
                state.activeId = index ? String(index) : null;
                renderOverlay();
            }
        };
    }

    const viewers = [];
    const inlineViewer = createViewer({
        name: "inline",
        stageId: "inlineViewerStage",
        canvasId: "inlineViewerCanvas",
        imageId: "inlineViewerImage",
        overlayId: "inlineViewerOverlay",
        tooltipId: "inlineViewerTooltip",
        infoCardId: "inlineViewerInfoCard",
        modeGroupId: null,
        enablePanZoom: false,
        defaultMode: "boxes"
    });

    const modalViewer = createViewer({
        name: "modal",
        stageId: "modalViewerStage",
        canvasId: "modalViewerCanvas",
        imageId: "modalViewerImage",
        overlayId: "modalViewerOverlay",
        tooltipId: "modalViewerTooltip",
        infoCardId: "modalViewerInfoCard",
        modeGroupId: null,
        enablePanZoom: true,
        emptyId: "modalViewerEmpty",
        defaultMode: "index"
    });

    if (inlineViewer) {
        viewers.push(inlineViewer);
        inlineViewer.mount();
    }
    if (modalViewer) {
        viewers.push(modalViewer);
        modalViewer.mount();
    }

    function updateGlobalSelection(index, sourceName) {
        const selectedId = index ? String(index) : null;

        detectionRows.forEach(function (row) {
            const isActive = row.dataset.detIndex === selectedId;
            row.classList.toggle("is-active", isActive);
            row.setAttribute("aria-selected", String(isActive));
        });

        viewers.forEach(function (viewer) {
            if (viewer.name !== sourceName) {
                viewer.setExternalActive(selectedId);
            }
        });
    }

    function bindTableRowEvents() {
        detectionRows.forEach(function (row) {
            const index = row.dataset.detIndex;

            row.addEventListener("click", function () {
                if (inlineViewer) {
                    inlineViewer.scrollIntoView();
                    inlineViewer.setExternalActive(index);
                }
                if (modalViewer && viewerModalBackdrop && !viewerModalBackdrop.hidden) {
                    modalViewer.setExternalActive(index);
                }
                updateGlobalSelection(index, "");
            });

            row.addEventListener("keydown", function (event) {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    row.click();
                }
            });
        });
    }

    function openViewerModal() {
        if (!viewerModalBackdrop) {
            return;
        }
        viewerModalBackdrop.hidden = false;
        document.body.classList.add("is-modal-open");
        if (modalViewer) {
            modalViewer.open();
        }
    }

    function closeViewerModal() {
        if (!viewerModalBackdrop) {
            return;
        }
        viewerModalBackdrop.hidden = true;
        document.body.classList.remove("is-modal-open");
        if (modalViewer) {
            modalViewer.close();
        }
    }

    if (imageInput && previewImage) {
        imageInput.addEventListener("change", function (event) {
            const file = event.target.files[0];
            if (!file) {
                return;
            }

            const reader = new FileReader();
            reader.onload = function (loadEvent) {
                previewImage.src = loadEvent.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    if (confRange) {
        confRange.addEventListener("input", updateRangeText);
    }
    if (iouRange) {
        iouRange.addEventListener("input", updateRangeText);
    }
    if (modelSelect) {
        modelSelect.addEventListener("change", updateModelInfo);
    }
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            window.location.href = "/";
        });
    }

    if (openViewerModalBtn) {
        openViewerModalBtn.addEventListener("click", openViewerModal);
    }
    if (viewerModalClose) {
        viewerModalClose.addEventListener("click", closeViewerModal);
    }
    if (viewerModalBackdrop) {
        viewerModalBackdrop.addEventListener("click", function (event) {
            if (event.target === viewerModalBackdrop) {
                closeViewerModal();
            }
        });
    }
    if (viewerResetBtn && modalViewer) {
        viewerResetBtn.addEventListener("click", function () {
            modalViewer.resetView();
        });
    }

    if (errorModalClose) {
        errorModalClose.addEventListener("click", closeErrorModal);
    }
    if (errorModalConfirm) {
        errorModalConfirm.addEventListener("click", closeErrorModal);
    }
    if (errorModalBackdrop) {
        errorModalBackdrop.addEventListener("click", function (event) {
            if (event.target === errorModalBackdrop) {
                closeErrorModal();
            }
        });
    }

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            closeErrorModal();
            closeViewerModal();
        }
    });

    bindTableRowEvents();
    updateRangeText();
    updateModelInfo();
    openErrorModal(ERROR_MESSAGE);
});
