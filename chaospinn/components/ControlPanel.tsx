"use client";

import { useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card';
import { Play, Pause, RefreshCw } from 'lucide-react';
import { updateSimulationParams } from '@/lib/api';

export default function ControlPanel() {
  const [isRunning, setIsRunning] = useState(false);
  const [params, setParams] = useState({
    timeStep: 0.01,
    spatialPoints: 128,
    learningRate: 0.001,
  });

  const handleParamChange = async (param: string, value: number) => {
    const newParams = { ...params, [param]: value };
    setParams(newParams);
    try {
      await updateSimulationParams(newParams);
    } catch (error) {
      console.error('Failed to update parameters:', error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button
          variant={isRunning ? "destructive" : "default"}
          onClick={() => setIsRunning(!isRunning)}
        >
          {isRunning ? <Pause className="mr-2" /> : <Play className="mr-2" />}
          {isRunning ? 'Stop' : 'Start'} Simulation
        </Button>
        <Button variant="outline">
          <RefreshCw className="mr-2" />
          Reset
        </Button>
      </div>

      <div className="grid gap-6">
        <div className="space-y-2">
          <Label>Time Step</Label>
          <Slider
            value={[params.timeStep]}
            min={0.001}
            max={0.1}
            step={0.001}
            onValueChange={([value]) => handleParamChange('timeStep', value)}
          />
          <Input
            type="number"
            value={params.timeStep}
            onChange={(e) => handleParamChange('timeStep', parseFloat(e.target.value))}
          />
        </div>

        <div className="space-y-2">
          <Label>Spatial Points</Label>
          <Slider
            value={[params.spatialPoints]}
            min={32}
            max={256}
            step={32}
            onValueChange={([value]) => handleParamChange('spatialPoints', value)}
          />
          <Input
            type="number"
            value={params.spatialPoints}
            onChange={(e) => handleParamChange('spatialPoints', parseInt(e.target.value))}
          />
        </div>

        <div className="space-y-2">
          <Label>Learning Rate</Label>
          <Slider
            value={[params.learningRate]}
            min={0.0001}
            max={0.01}
            step={0.0001}
            onValueChange={([value]) => handleParamChange('learningRate', value)}
          />
          <Input
            type="number"
            value={params.learningRate}
            onChange={(e) => handleParamChange('learningRate', parseFloat(e.target.value))}
          />
        </div>
      </div>
    </div>
  );
}