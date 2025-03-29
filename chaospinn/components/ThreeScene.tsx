"use client";

import { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

const WaveMesh = ({ simulationData }: { simulationData: any }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  // Custom shader for the wave visualization
  const shaderMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uData: { value: null },
    },
    vertexShader: `
      varying vec2 vUv;
      uniform float uTime;
      uniform sampler2D uData;
      
      void main() {
        vUv = uv;
        vec4 data = texture2D(uData, uv);
        vec3 pos = position;
        pos.z += data.r * 2.0;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `,
    fragmentShader: `
      varying vec2 vUv;
      uniform float uTime;
      
      void main() {
        vec3 color = vec3(0.5 + 0.5 * sin(vUv.x * 10.0 + uTime),
                         0.5 + 0.5 * sin(vUv.y * 10.0 + uTime),
                         0.5 + 0.5 * sin((vUv.x + vUv.y) * 5.0 + uTime));
        gl_FragColor = vec4(color, 1.0);
      }
    `,
  });

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });

  return (
    <mesh ref={meshRef}>
      <planeGeometry args={[10, 10, 128, 128]} />
      <primitive object={shaderMaterial} ref={materialRef} />
    </mesh>
  );
};

export default function ThreeScene({ simulationData }: { simulationData: any }) {
  return (
    <Canvas>
      <PerspectiveCamera makeDefault position={[0, 0, 5]} />
      <OrbitControls enableDamping dampingFactor={0.05} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <WaveMesh simulationData={simulationData} />
    </Canvas>
  );
}