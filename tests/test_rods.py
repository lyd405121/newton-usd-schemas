# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from pxr import Plug, Usd, UsdShade

import newton_usd_schemas  # noqa: F401

USD_HAS_LIMITS = Usd.GetVersion() >= (0, 25, 11)


class TestNewtonCurvesDeformableMaterialAPI(unittest.TestCase):
    def setUp(self):
        self.stage: Usd.Stage = Usd.Stage.CreateInMemory()
        self.material = UsdShade.Material.Define(self.stage, "/CableMaterial").GetPrim()

    def test_api_registered(self):
        plug_type = Plug.Registry().FindTypeByName("NewtonPhysicsCurvesDeformableMaterialAPI")
        self.assertEqual(plug_type.typeName, "NewtonPhysicsCurvesDeformableMaterialAPI")
        schema_type = Usd.SchemaRegistry().GetSchemaTypeName("NewtonPhysicsCurvesDeformableMaterialAPI")
        self.assertEqual(schema_type, "NewtonCurvesDeformableMaterialAPI")

    def test_api_application(self):
        self.assertFalse(self.material.HasAPI("NewtonCurvesDeformableMaterialAPI"))
        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        self.assertTrue(self.material.HasAPI("PhysicsMaterialAPI"))
        self.assertTrue(self.material.HasAPI("NewtonCurvesDeformableMaterialAPI"))

        for name in (
            "newton:youngsModulus",
            "newton:poissonRatio",
            "newton:stretchDamping",
            "newton:bendDamping",
            "newton:twistDamping",
        ):
            self.assertTrue(self.material.HasAttribute(name), name)

    def test_api_limitations(self):
        prim = self.stage.DefinePrim("/NotMaterial", "Xform")
        self.assertFalse(prim.CanApplyAPI("NewtonCurvesDeformableMaterialAPI"))
        self.assertTrue(self.material.CanApplyAPI("NewtonCurvesDeformableMaterialAPI"))

    def test_material_parameter_defaults(self):
        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        self.assertEqual(self.material.GetAttribute("newton:youngsModulus").Get(), -math.inf)
        self.assertEqual(self.material.GetAttribute("newton:poissonRatio").Get(), -math.inf)
        self.assertEqual(self.material.GetAttribute("newton:stretchDamping").Get(), -math.inf)
        self.assertEqual(self.material.GetAttribute("newton:bendDamping").Get(), -math.inf)
        self.assertEqual(self.material.GetAttribute("newton:twistDamping").Get(), -math.inf)

    def test_material_parameter_roundtrip(self):
        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")

        values = {
            "newton:youngsModulus": 1.2e7,
            "newton:poissonRatio": 0.35,
            "newton:stretchDamping": 0.01,
            "newton:bendDamping": 0.02,
            "newton:twistDamping": 0.03,
        }
        for name, value in values.items():
            attr = self.material.GetAttribute(name)
            self.assertTrue(attr.Set(value), name)
            self.assertAlmostEqual(attr.Get(), value)

    def test_limits(self):
        if not USD_HAS_LIMITS:
            self.skipTest("USD build does not expose schema limits")

        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        for name in (
            "newton:youngsModulus",
            "newton:stretchDamping",
            "newton:bendDamping",
            "newton:twistDamping",
        ):
            limits = self.material.GetAttribute(name).GetSoftLimits()
            self.assertTrue(limits.IsValid(), name)
            self.assertEqual(limits.GetMinimum(), 0.0)
            self.assertIsNone(limits.GetMaximum())

        poisson = self.material.GetAttribute("newton:poissonRatio").GetSoftLimits()
        self.assertTrue(poisson.IsValid())
        self.assertEqual(poisson.GetMinimum(), -1.0)
        self.assertEqual(poisson.GetMaximum(), 0.5)


if __name__ == "__main__":
    unittest.main()
