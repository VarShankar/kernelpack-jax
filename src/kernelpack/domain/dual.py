from __future__ import annotations

from dataclasses import dataclass, field

from .core import DomainDescriptor


@dataclass
class DualNodeDomainDescriptor:
    velocity_domain_: DomainDescriptor = field(default_factory=DomainDescriptor)
    pressure_domain_: DomainDescriptor = field(default_factory=DomainDescriptor)
    has_velocity_: bool = False
    has_pressure_: bool = False

    def set_velocity_domain(self, domain: DomainDescriptor) -> None:
        self.velocity_domain_ = domain
        self.has_velocity_ = True

    def set_pressure_domain(self, domain: DomainDescriptor) -> None:
        self.pressure_domain_ = domain
        self.has_pressure_ = True

    def build_structs(self) -> None:
        if self.has_velocity_:
            self.velocity_domain_.build_structs()
        if self.has_pressure_:
            self.pressure_domain_.build_structs()

    def has_velocity_domain(self) -> bool:
        return self.has_velocity_

    def has_pressure_domain(self) -> bool:
        return self.has_pressure_

    def get_velocity_domain(self) -> DomainDescriptor:
        if not self.has_velocity_:
            raise ValueError("DualNodeDomainDescriptor does not contain a velocity domain")
        return self.velocity_domain_

    def get_pressure_domain(self) -> DomainDescriptor:
        if not self.has_pressure_:
            raise ValueError("DualNodeDomainDescriptor does not contain a pressure domain")
        return self.pressure_domain_

    def get_dim(self) -> int:
        if self.has_velocity_:
            return self.velocity_domain_.get_dim()
        if self.has_pressure_:
            return self.pressure_domain_.get_dim()
        return 0
