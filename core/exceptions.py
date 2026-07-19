class EKYCError(Exception):
    pass


class NoFaceDetectedError(EKYCError):
    pass


class MultipleFacesDetectedError(EKYCError):
    pass


class InvalidVideoError(EKYCError):
    pass

class MediaLoadError(EKYCError):
    pass
