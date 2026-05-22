import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const qualitative = document.querySelector("[data-paqa-qualitative]");

if (qualitative) {
  const assetRoot = "/assets/models/3dpaqa/qualitative-result-bundle/";
  const criteria = [
    "geometry",
    "texture",
    "material",
    "plausibility",
    "artifacts",
    "preference",
  ];
  const tablist = qualitative.querySelector(".paqa-criterion-tabs");
  const grid = qualitative.querySelector(".paqa-qualitative-grid");
  const loader = new GLTFLoader();
  const renderModes = [
    { id: "pbr", label: "RGB" },
    { id: "normal", label: "Normal" },
    { id: "material", label: "Material map" },
  ];
  const criterionDefinitions = {
    geometry: {
      title: "Geometry quality",
      copy: "Assesses how accurately the 3D shape represents the intended object and how much meaningful geometric detail is present, using normal-map structure rather than texture appearance.",
    },
    texture: {
      title: "Texture quality",
      copy: "Assesses surface appearance from RGB and texture evidence, including color fidelity, visual detail, and high-frequency patterns that make the asset look well textured.",
    },
    material: {
      title: "Material quality",
      copy: "Assesses the material representation encoded by PBR properties such as metallic and roughness maps, including whether surface properties are differentiated and detailed.",
    },
    plausibility: {
      title: "3D plausibility",
      copy: "Assesses whether the object forms a coherent and believable 3D asset, with sensible structure, proportions, and parts for what it is meant to represent.",
    },
    artifacts: {
      title: "Artifact quality",
      copy: "Assesses whether visible defects are absent across geometry and appearance, including distortions, noise, broken surfaces, and unintended structures.",
    },
    preference: {
      title: "Overall preference",
      copy: "Assesses overall human preference by considering geometry, texture, material, plausibility, and artifacts together as a single quality judgment.",
    },
  };
  let viewers = [];

  const titleCase = (value) => value.charAt(0).toUpperCase() + value.slice(1);
  const assetUrl = (asset) => `${assetRoot}glbs/${asset.cell_glb_filename}`;
  const scoreLabel = (asset) =>
    `Ours: ${asset.criterion_vlm_score.toFixed(1)} / Human: ${asset.criterion_human_score.toFixed(1)}`;
  const definitionTitle = qualitative.querySelector(
    ".paqa-criterion-definition strong",
  );
  const definitionCopy = qualitative.querySelector(
    ".paqa-criterion-definition p",
  );

  const disposeMaterial = (material) => {
    Object.values(material).forEach((value) => {
      if (value?.isTexture) value.dispose();
    });
    material.dispose();
  };

  const disposeObject = (object) => {
    object.traverse((child) => {
      if (!child.isMesh) return;
      child.geometry?.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach(disposeMaterial);
        return;
      }
      if (child.material) disposeMaterial(child.material);
    });
  };

  const mountViewer = (canvas, status, url) => {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f8fa);

    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
    camera.position.set(1.7, 1.2, 2.1);

    const renderer = new THREE.WebGLRenderer({ antialias: true, canvas });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 0.08;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x8c96a6, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(3, 5, 4);
    scene.add(keyLight);

    let frameId;
    let model;
    let stopped = false;
    const renderMaterials = [];
    let activeRenderMode = renderModes[0].id;

    const resize = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();

    const render = () => {
      if (stopped) return;
      controls.update();
      renderer.render(scene, camera);
      frameId = window.requestAnimationFrame(render);
    };

    const eachMesh = (callback) => {
      model?.traverse((child) => {
        if (child.isMesh) callback(child);
      });
    };

    const rememberMaterials = () => {
      eachMesh((mesh) => {
        mesh.userData.paqaMaterials = Array.isArray(mesh.material)
          ? mesh.material
          : [mesh.material];
      });
    };

    const normalMaterial = (source) => {
      const material = new THREE.MeshNormalMaterial({
        flatShading: source.flatShading,
        side: source.side,
        skinning: source.skinning,
        morphTargets: source.morphTargets,
      });
      renderMaterials.push(material);
      return material;
    };

    const materialMap = (source) => {
      const map = source.metalnessMap || source.roughnessMap;
      const fallback = source.metalness ?? source.roughness ?? 0.65;
      const material = new THREE.MeshBasicMaterial({
        color: map ? 0xffffff : new THREE.Color(fallback, fallback, fallback),
        map,
        side: source.side,
        skinning: source.skinning,
        morphTargets: source.morphTargets,
      });
      renderMaterials.push(material);
      return material;
    };

    const materialForMode = (source, mode) => {
      if (mode === "normal") return normalMaterial(source);
      if (mode === "material") return materialMap(source);
      return source;
    };

    const setRenderMode = (mode) => {
      activeRenderMode = mode;
      eachMesh((mesh) => {
        const original = mesh.userData.paqaMaterials;
        const materials = original.map((material) =>
          materialForMode(material, mode),
        );
        mesh.material = Array.isArray(mesh.material) ? materials : materials[0];
      });
    };

    loader.load(
      url,
      (gltf) => {
        model = gltf.scene;
        const bounds = new THREE.Box3().setFromObject(model);
        const center = bounds.getCenter(new THREE.Vector3());
        const size = bounds.getSize(new THREE.Vector3());
        const radius = Math.max(size.x, size.y, size.z, 0.2);

        model.position.sub(center);
        scene.add(model);
        rememberMaterials();
        setRenderMode(activeRenderMode);
        camera.near = Math.max(radius / 100, 0.01);
        camera.far = radius * 100;
        camera.position.set(radius * 1.55, radius * 0.95, radius * 1.8);
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        controls.minDistance = radius * 0.55;
        controls.maxDistance = radius * 5;
        controls.update();
        status.hidden = true;
      },
      undefined,
      () => {
        status.textContent = "Model preview unavailable.";
      },
    );
    render();

    const dispose = () => {
      stopped = true;
      window.cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      if (model) {
        scene.remove(model);
        disposeObject(model);
      }
      renderer.dispose();
      renderMaterials.forEach((material) => material.dispose());
    };

    return {
      dispose,
      setRenderMode,
    };
  };

  const createSample = (asset) => {
    const sample = document.createElement("article");
    sample.className = "paqa-sample";
    sample.innerHTML = `
      <header>
        <strong>Rank ${asset.column}</strong>
        <span>${scoreLabel(asset)}</span>
      </header>
      <div class="paqa-model-frame">
        <canvas aria-label="${titleCase(asset.criterion)} qualitative sample ${asset.column}"></canvas>
        <span class="paqa-model-status">Loading 3D asset...</span>
        <span class="paqa-gesture-cue" title="Drag to rotate. Scroll or pinch to zoom." aria-label="Drag to rotate. Scroll or pinch to zoom.">
          <span class="paqa-gesture-orbit" aria-hidden="true"></span>
          <span class="paqa-gesture-mouse" aria-hidden="true"></span>
        </span>
      </div>
      <div class="paqa-sample-modes" role="group" aria-label="${titleCase(asset.criterion)} sample ${asset.column} render mode"></div>
    `;

    grid.append(sample);
    const canvas = sample.querySelector("canvas");
    const status = sample.querySelector(".paqa-model-status");
    const modeGroup = sample.querySelector(".paqa-sample-modes");
    const viewer = mountViewer(canvas, status, assetUrl(asset));

    renderModes.forEach(({ id, label }, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.setAttribute("aria-pressed", index === 0);
      button.addEventListener("click", () => {
        modeGroup.querySelectorAll("button").forEach((modeButton) => {
          modeButton.setAttribute("aria-pressed", modeButton === button);
        });
        viewer.setRenderMode(id);
      });
      modeGroup.append(button);
    });

    viewers.push(viewer);
  };

  const showCriterion = (assets, criterion) => {
    viewers.forEach((viewer) => viewer.dispose());
    viewers = [];
    grid.replaceChildren();
    definitionTitle.textContent = criterionDefinitions[criterion].title;
    definitionCopy.textContent = criterionDefinitions[criterion].copy;

    tablist.querySelectorAll("button").forEach((button) => {
      const selected = button.dataset.criterion === criterion;
      button.setAttribute("aria-selected", selected);
      button.tabIndex = selected ? 0 : -1;
    });

    assets
      .filter((asset) => asset.criterion === criterion)
      .sort((left, right) => left.column - right.column)
      .forEach(createSample);
  };

  fetch(`${assetRoot}qualitative_result_annotations.json`)
    .then((response) => {
      if (!response.ok) throw new Error("Manifest request failed.");
      return response.json();
    })
    .then(({ cells }) => {
      criteria.forEach((criterion) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.criterion = criterion;
        button.setAttribute("role", "tab");
        button.textContent = titleCase(criterion);
        button.addEventListener("click", () => showCriterion(cells, criterion));
        tablist.append(button);
      });
      showCriterion(cells, criteria[0]);
    })
    .catch(() => {
      grid.innerHTML =
        '<p class="paqa-model-error">Qualitative assets could not be loaded.</p>';
    });
}
